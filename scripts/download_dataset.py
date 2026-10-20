"""Download the original SQuAD 2.0 splits and audit them without modification."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "squad_v2"
OUTPUTS = ROOT / "outputs"
CACHE = ROOT / ".cache" / "huggingface"
SOURCE = "rajpurkar/squad_v2"
EXAMPLES_PER_CLASS = 3

# Keep downloads and caches inside the project, including when launched elsewhere.
os.environ["HF_HOME"] = str(CACHE)
os.environ["HF_DATASETS_CACHE"] = str(CACHE / "datasets")
os.environ["HF_HUB_CACHE"] = str(CACHE / "hub")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    """Replace a generated report only after its complete JSON has been written."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def python_inventory() -> dict:
    candidates = {Path(sys.executable), Path(sys.base_prefix) / Path(sys.executable).name}
    for name in ("python", "python3", "python3.12"):
        located = shutil.which(name)
        if located:
            candidates.add(Path(located))
    local_appdata = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    candidates.update((local_appdata / "Programs" / "Python").glob("Python*/python.exe"))
    candidates.update((local_appdata / "Python").glob("*/python.exe"))
    candidates.add(
        Path.home() / ".cache" / "codex-runtimes" / "codex-primary-runtime"
        / "dependencies" / "python" / "python.exe"
    )
    if os.name == "nt":
        candidates.update(Path("C:/").glob("Python*/python.exe"))
        candidates.update(Path("C:/Program Files").glob("Python*/python.exe"))
    available, skipped = [], []
    for candidate in sorted(candidates, key=str):
        if not candidate.is_file():
            continue
        # Microsoft Store aliases can open an installer instead of running Python.
        if "windowsapps" in str(candidate).lower():
            skipped.append({"executable": str(candidate), "reason": "Windows Store alias"})
            continue
        try:
            result = subprocess.run(
                [str(candidate), "-c", "import platform; print(platform.python_version())"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            available.append({"executable": str(candidate), "version": result.stdout.strip()})
        except (OSError, subprocess.SubprocessError) as exc:
            skipped.append({"executable": str(candidate), "reason": str(exc)})
    return {
        "available": available,
        "skipped_or_unavailable": skipped,
        "scope": "Current interpreter, base interpreter, PATH and common local installation paths; not a full disk scan.",
    }


def main() -> int:
    started_at = utc_now()
    try:
        import datasets
        from datasets import DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        print(f"ERROR: datasets is unavailable: {exc}. Install requirements.txt with this project's .venv Python.", file=sys.stderr)
        return 1

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    provenance_path = DATA_PATH.parent / "squad_v2.provenance.json"
    stage = "loading the dataset"
    try:
        roundtrip = {"performed": False, "reason": "Existing local DatasetDict loaded directly."}
        if DATA_PATH.exists():
            mode = "local"
            print(f"Loading local DatasetDict with load_from_disk: {DATA_PATH}", flush=True)
            dataset = load_from_disk(str(DATA_PATH))
        else:
            mode = "download"
            print(f"Downloading real dataset with load_dataset({SOURCE!r})", flush=True)
            dataset = load_dataset(SOURCE)
            downloaded_at = utc_now()
            if not isinstance(dataset, DatasetDict):
                raise TypeError("The source did not return a DatasetDict.")
            stage = "saving and reopening the original DatasetDict"
            temporary = Path(tempfile.mkdtemp(prefix=".squad_v2.incomplete-", dir=DATA_PATH.parent))
            print(f"Saving the original splits to {DATA_PATH}", flush=True)
            dataset.save_to_disk(str(temporary))
            reopened = load_from_disk(str(temporary))
            if list(dataset) != list(reopened):
                raise ValueError("The split names changed after save_to_disk/load_from_disk.")
            split_checks = {}
            for name in dataset:
                before, after = dataset[name], reopened[name]
                split_checks[name] = {
                    "same_rows": len(before) == len(after),
                    "same_columns": before.column_names == after.column_names,
                    "same_features": before.features == after.features,
                    "same_arrow_data": before.data.table.equals(after.data.table),
                }
                if not all(split_checks[name].values()):
                    raise ValueError(f"Saved data differ from source in split {name}.")
            # Publish only a complete, reopened copy. Never overwrite existing raw data.
            del reopened
            temporary.rename(DATA_PATH)
            write_json(provenance_path, {"source_id": SOURCE, "downloaded_at_utc": downloaded_at})
            dataset = load_from_disk(str(DATA_PATH))
            roundtrip = {"performed": True, "passed": True, "splits": split_checks}

        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"{DATA_PATH} is not a DatasetDict; existing files were left unchanged.")
        loaded_at = utc_now()
        provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else None
        stage = "checking the original splits"
        split_summaries, split_ids, examples = {}, {}, []
        for name, split in dataset.items():
            required = {"id", "context", "question", "answers"}
            if not required.issubset(split.column_names):
                raise ValueError(f"Split {name} lacks fields: {sorted(required - set(split.column_names))}")
            ids = Counter()
            answerable_count = empty_context = empty_question = 0
            selected = {True: 0, False: 0}
            for row in split:
                answerable = len(row["answers"]["text"]) > 0
                answerable_count += int(answerable)
                empty_context += int(not row["context"].strip())
                empty_question += int(not row["question"].strip())
                ids[row["id"]] += 1
                if selected[answerable] < EXAMPLES_PER_CLASS:
                    examples.append({
                        "split": name,
                        **{key: row[key] for key in ("id", "context", "question", "answers")},
                        "answerable": answerable,
                    })
                    selected[answerable] += 1
            split_ids[name] = set(ids)
            split_summaries[name] = {
                "rows": len(split),
                "columns": split.column_names,
                "answerable": answerable_count,
                "unanswerable": len(split) - answerable_count,
                "empty_context": empty_context,
                "empty_question": empty_question,
                "unique_ids": len(ids),
                "duplicate_id_values": sum(count > 1 for count in ids.values()),
                "duplicate_id_extra_rows": sum(count - 1 for count in ids.values()),
                "fingerprint": getattr(split, "_fingerprint", None),
            }
            print(f"{name}: {json.dumps(split_summaries[name], ensure_ascii=False)}", flush=True)
        intersections = [
            {"splits": [left, right], "shared_id_count": len(split_ids[left] & split_ids[right])}
            for left, right in combinations(dataset, 2)
        ]
        summary = {
            "source_id": SOURCE,
            "dataset_path": str(DATA_PATH),
            "started_at_utc": started_at,
            "downloaded_at_utc": provenance.get("downloaded_at_utc") if provenance else None,
            "loaded_at_utc": loaded_at,
            "checked_at_utc": utc_now(),
            "load_mode": mode,
            "environment": {
                "os": platform.platform(),
                "python_version": platform.python_version(),
                "python_executable": sys.executable,
                "datasets_version": datasets.__version__,
                "python_inventory": python_inventory(),
            },
            "answerable_definition": "len(answers['text']) > 0",
            "empty_text_definition": "not text.strip(); original text is preserved",
            "splits": split_summaries,
            "id_intersections": intersections,
            "save_load_roundtrip": roundtrip,
        }
        # Preserve the first run's actual disk round-trip evidence on local reruns.
        summary_path = OUTPUTS / "dataset_summary.json"
        if mode == "local" and summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            previous_check = previous.get("initial_save_load_roundtrip", previous.get("save_load_roundtrip"))
            if previous_check and previous_check.get("performed"):
                summary["initial_save_load_roundtrip"] = previous_check
        write_json(OUTPUTS / "dataset_examples.json", {"source_id": SOURCE, "examples": examples})
        write_json(summary_path, summary)
        print(f"ID intersections: {json.dumps(intersections)}", flush=True)
        print(f"Done. load_mode={mode}; reports: {OUTPUTS}", flush=True)
        return 0
    except Exception as exc:
        print(
            f"ERROR while {stage}: {type(exc).__name__}: {exc}\n"
            "Check network access to Hugging Face, free disk space and file permissions. "
            "An existing local copy must be a complete save_to_disk DatasetDict. "
            "No synthetic data were created; existing raw data were not overwritten. "
            "A failed save may leave a .squad_v2.incomplete-* directory for inspection.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
