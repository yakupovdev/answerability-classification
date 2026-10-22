"""Inspect local Windows hardware and project libraries without installing anything."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from ctypes import wintypes
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def windows_hardware() -> dict:
    import winreg

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    cpu = {"logical_processors": os.cpu_count()}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu["name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError as exc:
        cpu["name_error"] = str(exc)

    topology = kernel.GetLogicalProcessorInformationEx
    topology.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    topology.restype = wintypes.BOOL
    size = wintypes.DWORD()
    topology(0, None, ctypes.byref(size))  # RelationProcessorCore
    if size.value:
        buffer = ctypes.create_string_buffer(size.value)
        if topology(0, buffer, ctypes.byref(size)):
            offset = cores = 0
            while offset < size.value:
                relationship = wintypes.DWORD.from_buffer(buffer, offset).value
                entry_size = wintypes.DWORD.from_buffer(buffer, offset + 4).value
                if entry_size < 8 or offset + entry_size > size.value:
                    raise RuntimeError("Invalid Windows CPU topology response.")
                cores += int(relationship == 0)
                offset += entry_size
            cpu["physical_cores"] = cores
        else:
            cpu["physical_cores_error"] = str(ctypes.WinError(ctypes.get_last_error()))
    else:
        cpu["physical_cores_error"] = str(ctypes.WinError(ctypes.get_last_error()))
    cpu["method"] = "Windows registry ProcessorNameString; GetLogicalProcessorInformationEx; os.cpu_count"

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "avail_phys", "total_pagefile", "avail_pagefile",
                "total_virtual", "avail_virtual", "avail_extended_virtual",
            )
        ]

    ram = {}
    installed_kib = ctypes.c_ulonglong()
    installed = kernel.GetPhysicallyInstalledSystemMemory
    installed.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    installed.restype = wintypes.BOOL
    if installed(ctypes.byref(installed_kib)):
        ram["installed_bytes"] = installed_kib.value * 1024
        ram["installed_gib"] = ram["installed_bytes"] / 1024**3
    else:
        ram["installed_memory_error"] = str(ctypes.WinError(ctypes.get_last_error()))
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    memory_status = kernel.GlobalMemoryStatusEx
    memory_status.argtypes = [ctypes.POINTER(MemoryStatus)]
    memory_status.restype = wintypes.BOOL
    if memory_status(ctypes.byref(status)):
        ram.update({
            "os_usable_total_bytes": status.total_phys,
            "os_usable_total_gib": status.total_phys / 1024**3,
            "available_bytes_at_check": status.avail_phys,
            "available_gib_at_check": status.avail_phys / 1024**3,
        })
    else:
        ram["usable_memory_error"] = str(ctypes.WinError(ctypes.get_last_error()))
    ram["method"] = "GetPhysicallyInstalledSystemMemory (installed); GlobalMemoryStatusEx (OS usable/available)"
    return {"cpu": cpu, "ram": ram}


def check_nvidia() -> dict:
    executable = shutil.which("nvidia-smi")
    result = {"executable": executable, "status": "not_found", "gpus": []}
    result["cuda_version_explanation"] = (
        "CUDA Version (called CUDA UMD Version in this nvidia-smi release) is the latest CUDA "
        "version supported by the driver. It does not prove that CUDA Toolkit is installed "
        "or that the project's PyTorch can use CUDA."
    )
    result["reference"] = "https://docs.nvidia.com/deploy/nvidia-smi/index.html#cuda-umd-version"
    result["cuda_toolkit_installation"] = "Not checked; cannot be inferred from nvidia-smi."
    if not executable:
        result["reason"] = "nvidia-smi is not on PATH."
        return result
    try:
        header = subprocess.run([executable], capture_output=True, text=True, timeout=30, check=False)
        result["header_exit_code"] = header.returncode
        result["header_stdout"] = header.stdout
        result["header_stderr"] = header.stderr
        match = re.search(r"CUDA(?: UMD)? Version:\s*([\d.]+)", header.stdout)
        result["driver_supported_cuda_version"] = match.group(1) if match else None
        query = subprocess.run(
            [executable, "--query-gpu=index,name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        result.update({"query_exit_code": query.returncode, "query_stdout": query.stdout, "query_stderr": query.stderr})
        if query.returncode != 0:
            result["status"] = "query_failed"
            return result
        for row in csv.reader(query.stdout.splitlines(), skipinitialspace=True):
            index, name, total, free, driver = [value.strip() for value in row]
            result["gpus"].append({
                "index": int(index), "name": name, "memory_total_mib": int(total),
                "memory_free_mib": int(free), "driver_version": driver,
            })
        result["status"] = "ok"
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return result


def check_libraries() -> dict:
    result = {}
    for distribution, module in {"torch": "torch", "transformers": "transformers", "scikit-learn": "sklearn", "accelerate": "accelerate"}.items():
        try:
            installed_version = version(distribution)
        except PackageNotFoundError:
            installed_version = None
        result[distribution] = {
            "installed": installed_version is not None,
            "version": installed_version,
            "module_found": importlib.util.find_spec(module) is not None,
        }
    return result


def check_torch(libraries: dict) -> dict:
    result = {
        "status": "skipped", "torch_version": None, "torch_version_cuda": None,
        "cuda_is_available": None, "gpu_names": [],
        "gpu_tensor_operation": {"status": "skipped", "reason": "PyTorch is not installed in the project environment."},
    }
    if not libraries["torch"]["module_found"]:
        result["reason"] = "PyTorch is not installed; this is not a project error."
        return result
    try:
        import torch

        result.update({"status": "checked", "torch_version": torch.__version__,
                       "torch_version_cuda": torch.version.cuda, "cuda_is_available": torch.cuda.is_available()})
        if not result["cuda_is_available"]:
            result["gpu_tensor_operation"]["reason"] = "torch.cuda.is_available() returned False."
            return result
        result["gpu_names"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        result["gpu_tensor_operation"] = {"status": "running", "operation": "64x64 float32 ones matrix multiplication; all results must equal 64", "devices": []}
        with torch.no_grad():
            for index, name in enumerate(result["gpu_names"]):
                device = torch.device(f"cuda:{index}")
                tensor = torch.ones((64, 64), dtype=torch.float32, device=device)
                product = tensor @ tensor
                torch.cuda.synchronize(device)
                if not bool(torch.all(product == 64).item()):
                    raise RuntimeError("GPU tensor result validation failed.")
                result["gpu_tensor_operation"]["devices"].append({"index": index, "name": name, "passed": True})
        result["gpu_tensor_operation"]["status"] = "passed"
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        if result["gpu_tensor_operation"]["status"] == "running":
            result["gpu_tensor_operation"]["status"] = "failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, help="Defaults to a new outputs/audits/<UTC>_environment directory.")
    args = parser.parse_args()
    report_dir = args.report_dir or ROOT / "outputs" / "audits" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_environment")
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_path = report_dir / "environment_check.json"
    if report_path.exists():
        raise ValueError("Environment report already exists; choose a new --report-dir.")
    inputs = [ROOT / "README.md", ROOT / "configs" / "data_split.json",
              ROOT / "configs" / "classifier_distilbert.json"]
    split_summary = ROOT / "outputs" / "split_summary.json"
    if split_summary.is_file():
        inputs.append(split_summary)
    input_checksums = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    libraries = check_libraries()
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "platform": platform.platform(),
        "python": {"version": platform.python_version(), "full_version": sys.version, "executable": sys.executable, "is_venv": sys.prefix != sys.base_prefix},
        "hardware": windows_hardware(),
        "nvidia_smi": check_nvidia(),
        "libraries": libraries,
        "pytorch": check_torch(libraries),
        "input_document_sha256": input_checksums,
        "scope": "Hardware and environment inspection only. No dataset opened, no test examples used, no packages installed, no models downloaded or trained.",
    }
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != input_checksums[str(path.relative_to(ROOT))] for path in inputs):
        raise RuntimeError("Project configuration/documentation changed during inspection.")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Environment report: {report_path}")


if __name__ == "__main__":
    main()
