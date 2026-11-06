"""Prepare and run a paired local LLM comparison; only 'test' opens the held-out test dataset."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import traceback
from importlib.metadata import version
from pathlib import Path

from comparison_utils import (ROOT, SCORING, EVALUATION_POLICY, append_jsonl, check_ids, check_resume,
                              cluster_bootstrap, digest, file_sha, gate_replay, ids_checksum,
                              paired_report, parse_output, project_path, read_json, read_jsonl,
                              relative, require, review_candidates, tree_sha, utc_now,
                              validate_reviews, write_json, score_row, review_bundle,
                              assessment_html, expand_review_bundle)
from report_results import completed_run

OUTPUTS = ROOT / "outputs" / "llm_comparison"
PROCESSED = ROOT / "data" / "processed" / "squad_v2_answerability"
CONFIG = ROOT / "configs" / "llm_comparison.json"
SELECTION_RULE = "Maximum validation macro-F1 among completed full 3-epoch classifier runs with seed 42/43/44 and verified export; ties: smaller seed, then lexicographic run_id for the same seed"


def versions():
    return {"python": platform.python_version(), **{name: version(name) for name in
            ("torch", "transformers", "datasets", "huggingface_hub", "accelerate", "tokenizers", "numpy", "scikit-learn", "safetensors", "pyarrow")}}


def configure_cache(*, offline=True):
    os.environ.update(HF_HOME=str(ROOT / ".cache" / "huggingface"),
                      HF_HUB_CACHE=str(ROOT / ".cache" / "huggingface" / "hub"),
                      HF_HUB_DISABLE_TELEMETRY="1", HF_DATASETS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ["HF_HUB_OFFLINE"] = "0"


def read_config(path):
    c = read_json(path)
    expected = set(read_json(CONFIG))
    require(set(c) == expected, "Unexpected/missing comparison config keys")
    require(c["batch_size"] == 1 and c["device"] == "cuda:0" and c["attention"] == "sdpa",
            "This implementation uses batch=1, cuda:0 and sdpa, without quantization/offload")
    require(c["dtype"] in ("bfloat16", "float16", "float32") and c["classifier_dtype"] == "float32",
            "Choose explicit generator precision; classifier scoring remains FP32")
    require(c["do_sample"] is False and c["num_beams"] == 1 and c["use_cache"] is True,
            "Only greedy decoding with KV cache is supported")
    require(c["threshold"] == .5 and c["classifier_max_length"] == 512,
            "Main config uses threshold=0.5 and classifier max_length=512; threshold studies require the explicit validation command")
    require(type(c["max_new_tokens"]) is int and type(c["max_sequence_tokens"]) is int
            and 0 < c["max_new_tokens"] < c["max_sequence_tokens"], "Invalid token budget")
    require(c["bootstrap_repeats"] == 2000 and c["subset_seed"] == 42, "Use 2000 bootstrap replicates and fixed subset seed=42")
    for key in ("smoke_rows", "development_rows", "bootstrap_seed", "inference_seed"):
        require(type(c[key]) is int and c[key] > 0, f"Invalid {key}")
    require(c["smoke_rows"] < c["development_rows"], "Smoke must be smaller than development")
    require("NO_ANSWER" in c["system_prompt"], "System prompt must specify NO_ANSWER")
    # Reject accidental prompt fields such as ID, gold, label or metrics.
    import string
    fields = [field for _, field, spec, conv in string.Formatter().parse(c["user_prompt"]) if field is not None]
    require(sorted(fields) == ["context", "question"], "User prompt may contain only question and context, once each")
    return c


def candidates():
    accepted, excluded = [], []
    reference = read_json(ROOT / "configs" / "data_split.json")
    for directory in sorted((ROOT / "outputs" / "classifier").glob("*")):
        if not directory.is_dir():
            continue
        try:
            run = completed_run(directory)
            info = run["info"]
            require(info["config"]["seed"] in (42, 43, 44) and info["config"]["epochs"] == 3,
                    "Requires seed 42/43/44 and 3 epochs")
            require(info["data"]["ordered_ids_sha256"]["validation"] == reference["splits"]["validation"]["ids_sha256"],
                    "Different validation identity")
            accepted.append({"run_id": directory.name, "run_path": relative(directory),
                             "seed": info["config"]["seed"], "macro_f1": run["final"]["model"]["macro_f1"],
                             "export_passed": info["export"]["passed"],
                             "model_path": relative(project_path(info["model_directory"])),
                             "model_exists": project_path(info["model_directory"]).is_dir(),
                             "report_sha256": {name: file_sha(directory / name) for name in
                                               ("run.json", "validation_metrics.json", "epoch_metrics.jsonl")}})
        except (ValueError, KeyError, FileNotFoundError) as exc:
            excluded.append({"run_id": directory.name, "reason": str(exc)})
    return sorted(accepted, key=lambda r: (-r["macro_f1"], r["seed"], r["run_id"])), excluded


def list_classifiers():
    runs, excluded = candidates()
    print(SELECTION_RULE)
    for i, run in enumerate(runs):
        print(f"{'SELECTED' if i == 0 else 'candidate':9} {run['run_id']} seed={run['seed']} "
              f"macro-F1={run['macro_f1']:.9f} export={run['export_passed']} model_exists={run['model_exists']} model={run['model_path']}")
    for run in excluded:
        print(f"EXCLUDED {run['run_id']}: {run['reason']}")
    require(runs, "No eligible completed full classifier runs found")


def classifier_identity(run_path, model_path):
    pool, _ = candidates()
    require(pool, "No eligible classifiers")
    chosen = pool[0]
    require(relative(run_path) == chosen["run_path"] and relative(model_path) == chosen["model_path"],
            f"Explicit classifier paths violate selection rule; selected run is {chosen['run_id']}")
    require(chosen["model_exists"], "Selected export is missing; restore it, do not silently select a worse run")
    info = read_json(run_path / "run.json")
    validation_files = tree_sha(PROCESSED / "validation")
    require(validation_files and validation_files == {
        key.removeprefix("validation/"): value for key, value in info["data"]["files_sha256"].items()
        if key.startswith("validation/")}, "Validation files differ from the classifier's recorded training/evaluation data")
    for other in pool:
        other_info = read_json(project_path(other["run_path"]) / "run.json")
        require({k: v for k, v in other_info["config"].items() if k != "seed"}
                == {k: v for k, v in info["config"].items() if k != "seed"}, "Classifier candidates differ beyond seed")
    files = tree_sha(model_path)
    require("config.json" in files and "tokenizer_config.json" in files and "tokenizer.json" in files
            and any(key.endswith(".safetensors") for key in files), "Classifier export is incomplete")
    model_config = read_json(model_path / "config.json")
    require(model_config.get("id2label") == {"0": "unanswerable", "1": "answerable"},
            "Classifier class order must be 0=unanswerable, 1=answerable")
    return {"selection_rule": SELECTION_RULE, "selected": chosen, "candidates": pool,
            "base_model": {key: info["model_source"][key] for key in ("model_id", "revision", "tokenizer_revision")}, "files_sha256": files,
            "validation_files_sha256": validation_files,
            "tokenizer_settings": read_json(model_path / "tokenizer_config.json"),
            "input": {"order": ["question", "context"], "max_length": 512, "truncation": "only_second", "dtype": "float32"}}


def tokenizer_identity(tokenizer):
    return {"class": type(tokenizer).__name__, "is_fast": tokenizer.is_fast,
            "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
            "special_tokens_map": tokenizer.special_tokens_map,
            "chat_template": tokenizer.chat_template}


def prepare_generator(args):
    destination = project_path(args.output)
    require(not destination.exists(), "Generator lock already exists; choose a new --output")
    c = read_config(project_path(args.config))
    configure_cache(offline=False)
    # Load PyTorch first so its native dependencies are registered on Windows.
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    sha = HfApi().model_info(c["model_id"], revision=args.revision).sha
    require(re.fullmatch("[0-9a-f]{40}", sha or ""), "Hub did not return an immutable commit SHA")
    snapshot = Path(snapshot_download(c["model_id"], revision=sha,
                                     cache_dir=str(ROOT / ".cache" / "huggingface" / "hub"),
                                     allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "*.jinja"]))
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    require(model_config.model_type == "qwen2", "Expected Qwen2 architecture")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer.chat_template and tokenizer.is_fast, "Official fast tokenizer and chat template are required")
    files = tree_sha(snapshot)
    require(any(key.endswith(".safetensors") for key in files), "No safetensors downloaded")
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        require(set(read_json(index)["weight_map"].values()).issubset(files), "Missing generator weight shards")
    write_json(destination, {"schema": "local-generator-lock-v1", "created_at_utc": utc_now(),
                             "model_id": c["model_id"], "requested_revision": args.revision,
                             "model_revision": sha, "tokenizer_revision": sha,
                             "snapshot_path": relative(snapshot), "files_sha256": files,
                             "tokenizer": tokenizer_identity(tokenizer),
                             "max_position_embeddings": model_config.max_position_embeddings,
                             "versions": versions(), "gpu_inference_checked": False})
    print(f"Generator and tokenizer locked to {sha}\nLock: {destination}\nNo model inference performed.")


def load_generator_lock(path, c):
    lock = read_json(path)
    require(lock["schema"] == "local-generator-lock-v1" and lock["model_id"] == c["model_id"], "Generator lock/model mismatch")
    require(lock["model_revision"] == lock["tokenizer_revision"]
            and re.fullmatch("[0-9a-f]{40}", lock["model_revision"]), "Invalid pinned generator/tokenizer revision")
    snapshot = project_path(lock["snapshot_path"])
    require(tree_sha(snapshot) == lock["files_sha256"], "Generator files changed/missing; restore the pinned cache")
    require(c["max_sequence_tokens"] <= lock["max_position_embeddings"], "Sequence budget exceeds generator capacity")
    return lock


def code_signature():
    return {relative(p): file_sha(p) for p in (Path(__file__).resolve(), ROOT / "scripts" / "comparison_utils.py",
            ROOT / "scripts" / "report_results.py", ROOT / "requirements.txt")}


def reference_identity():
    path = ROOT / "configs" / "data_split.json"
    reference = read_json(path)
    require(reference["source_id"] == "rajpurkar/squad_v2" and reference["settings"]["seed"] == 42,
            "Comparison requires the existing SQuAD 2.0 split protocol with split seed=42")
    return {"reference": reference, "reference_file_sha256": file_sha(path)}


def descriptor(config_path, classifier_run, classifier_model, generator_lock):
    c = read_config(config_path)
    return {"schema": "paired-llm-settings-v1", "config": c,
            "classifier": classifier_identity(classifier_run, classifier_model),
            "generator": load_generator_lock(generator_lock, c),
            "data": reference_identity(), "versions": versions(), "code_sha256": code_signature(),
            "scoring": SCORING, "method": "paired_replay", "no_generator_training": True}


def load_rows(split, expected_reference):
    require(split in ("validation", "test"), "Unknown evaluation split")
    configure_cache()
    from datasets import load_from_disk

    # Load a single split directory, never the entire DatasetDict.
    dataset = load_from_disk(str(PROCESSED / split))
    ids = list(dataset["id"])
    reference = expected_reference["reference"]["splits"][split]
    require(len(ids) == reference["rows"] and len(set(ids)) == len(ids)
            and ids_checksum(ids) == reference["ids_sha256"], f"{split} size/IDs/order differ from fixed split reference")
    manifest = read_json(ROOT / "outputs" / "split_manifest.json")["splits"][split]
    require(manifest["ids"] == ids and manifest["ids_sha256"] == ids_checksum(ids), f"{split} differs from manifest")
    rows = []
    import unicodedata
    import hashlib
    for row in dataset:
        require(row["label"] in (0, 1) and row["label"] == int(bool(row["answers"]["text"])), "Inconsistent answerability label")
        require(row["question"].strip() and row["context"].strip(), "Empty input")
        group = hashlib.sha256(" ".join(unicodedata.normalize("NFC", row["context"]).split()).encode()).hexdigest()
        require(group == row["context_group"], "Invalid context group")
        rows.append({key: row[key] for key in ("id", "question", "context", "answers", "label", "context_group")})
    if split == "test":
        require(len(rows) == 1000 and sum(row["label"] for row in rows) == 500, "Final test must be exactly 500+500 examples")
    return rows, {"split": split, "rows": len(rows), "ids_sha256": ids_checksum(ids),
                  "content_sha256": digest(rows), "files_sha256": tree_sha(PROCESSED / split),
                  "fingerprint": getattr(dataset, "_fingerprint", None)}


def validation_plan(settings, purpose):
    rows, signature = load_rows("validation", settings["data"])
    n = settings["config"]["smoke_rows" if purpose == "smoke" else "development_rows"]
    require(0 < n <= len(rows), "Invalid validation subset size")
    indices = sorted(random.Random(settings["config"]["subset_seed"]).sample(range(len(rows)), n))
    subset = [rows[i] for i in indices]
    return subset, {"source": signature, "algorithm": "random.Random(42).sample indices without replacement, sort sampled indices; preserve validation row order, no class balancing",
                    "purpose": purpose, "ids": [r["id"] for r in subset],
                    "ids_sha256": ids_checksum(r["id"] for r in subset), "content_sha256": digest(subset)}


def prompt_input(tokenizer, question, context, c):
    """Fit only context; never pass gold, ID, label or metric fields to the template."""
    budget = c["max_sequence_tokens"] - c["max_new_tokens"]
    def render(visible):
        messages = [{"role": "system", "content": c["system_prompt"]},
                    {"role": "user", "content": c["user_prompt"].format(question=question, context=visible)}]
        # Transformers 5.20 defaults to a dict; request the exact token-ID list explicitly.
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    full = render(context)
    require(len(render("")) <= budget, "Question and instructions exceed the input budget; context-only truncation cannot solve this")
    spans = tokenizer(context, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    selected = full
    kept = len(spans)
    if len(full) > budget:
        # Every accepted prefix is re-templated and measured, rather than slicing instructions/question.
        lo, hi = 0, len(spans)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            candidate = render(context[:spans[mid - 1][1]] if mid else "")
            if len(candidate) <= budget:
                lo = mid
            else:
                hi = mid - 1
        kept = lo
        selected = render(context[:spans[kept - 1][1]] if kept else "")
    require(len(selected) <= budget, "Context fitting failed")
    return selected, {"input_tokens": len(selected), "full_input_tokens": len(full), "input_budget": budget,
                      "original_context_tokens": len(spans), "visible_context_tokens": kept,
                      "visible_context_char_end": (spans[kept - 1][1] if kept else 0) if len(full) > budget else len(context),
                      "context_truncated": len(full) > budget, "question_preserved": True,
                      "input_ids_sha256": digest(selected)}


def gpu_setup(c, *, classifier=False):
    import torch

    require(torch.cuda.is_available(), "PyTorch CUDA is unavailable; no CPU fallback is performed")
    if not classifier and c["dtype"] == "bfloat16":
        require(torch.cuda.is_bf16_supported(), "BF16 is unsupported; explicitly choose a new config and revalidate, no precision fallback")
    torch.manual_seed(c["inference_seed"])
    torch.cuda.manual_seed_all(c["inference_seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    return {"gpu": torch.cuda.get_device_name(0), "total_bytes": props.total_memory,
            "cuda_build": torch.version.cuda, "precision": "float32" if classifier else c["dtype"],
            "deterministic_algorithms": True, "tf32": False, "inference_seed": c["inference_seed"]}


def latest_phase(directory, phase):
    paths = list(directory.glob(f"{phase}_phase_*.json"))
    require(paths, f"Missing actual {phase} phase report in {directory}")
    return read_json(max(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1])))


def check_frozen_runtime(meta, phase, actual):
    frozen = meta["identity"]["protocol"]
    if frozen is not None:
        path = project_path(frozen["path"])
        require(file_sha(path) == frozen["sha256"], "Frozen protocol changed during execution")
        require(read_json(path)[phase + "_runtime"] == actual,
                f"{phase} hardware/precision/tokenizer/generation settings differ from validated protocol")


def classifier_input(tokenizer, question, context):
    return tokenizer(question, context, max_length=512, truncation="only_second",
                     return_token_type_ids=False, return_tensors="pt", return_offsets_mapping=True)


def worker_classifier(directory, meta):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    settings, c = meta["identity"]["settings"], meta["identity"]["settings"]["config"]
    gpu = gpu_setup(c, classifier=True)
    path = project_path(settings["classifier"]["selected"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer.truncation_side == "right", "Classifier must retain the context prefix")
    check_frozen_runtime(meta, "classifier", {**gpu, "tokenizer": tokenizer_identity(tokenizer)})
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                             dtype=torch.float32).to(c["device"]).eval()
    rows = read_jsonl(directory / "examples.jsonl")
    saved = read_jsonl(directory / "classifier_scores.jsonl")
    check_ids(saved, [r["id"] for r in rows], complete=False)
    with torch.inference_mode():
        for row in rows[len(saved):]:
            encoded = classifier_input(tokenizer, row["question"], row["context"])
            sequence_ids = encoded.sequence_ids(0)
            offsets = encoded.pop("offset_mapping")[0].tolist()
            ends = [end for (start, end), sid in zip(offsets, sequence_ids) if sid == 1 and end > start]
            visible_end = max(ends, default=0)
            tokens = encoded["input_ids"][0].tolist()
            require(set(encoded) == {"input_ids", "attention_mask"}, "Unexpected classifier model input fields")
            tensors = {key: value.to(c["device"]) for key, value in encoded.items()}
            p = float(torch.softmax(model(**tensors).logits.float(), dim=-1)[0, 1].item())
            require(math.isfinite(p) and 0 <= p <= 1, "Nonfinite classifier probability")
            append_jsonl(directory / "classifier_scores.jsonl", {
                "id": row["id"], "p_answerable": p, "input_tokens": len(tokens),
                "visible_context_char_end": visible_end,
                "context_truncated": bool(row["context"][visible_end:].strip()),
                "input_ids_sha256": digest(tokens), "classifier_identity": digest(settings["classifier"])})
            print(f"classifier {row['id']}: p={p:.6f}", flush=True)
    torch.cuda.synchronize()
    write_json(directory / f"classifier_phase_{meta['attempt']}.json", {
        **gpu, "tokenizer": tokenizer_identity(tokenizer), "completed_rows": len(rows),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def ensure_filtered(directory, rows, baseline, scores, settings, threshold):
    saved = read_jsonl(directory / "filtered_predictions.jsonl")
    check_ids(saved, [r["id"] for r in rows], complete=False)
    require(len(saved) <= len(baseline), "Filtered output exists without its paired baseline")
    for i, b in enumerate(baseline):
        require(b["generator_identity"] == digest(settings), "Baseline generator/input configuration changed; replay refused")
        expected = gate_replay(b, scores[i]["p_answerable"], threshold)
        if i < len(saved):
            require(saved[i] == expected, "Existing filtered prediction differs from deterministic gate replay")
        else:
            append_jsonl(directory / "filtered_predictions.jsonl", expected)


def worker_generator(directory, meta):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    settings = meta["identity"]["settings"]
    c = settings["config"]
    gpu = gpu_setup(c)
    path = project_path(settings["generator"]["snapshot_path"])
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    require(tokenizer_identity(tokenizer) == settings["generator"]["tokenizer"], "Generator tokenizer/template settings changed")
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                              dtype=getattr(torch, c["dtype"]), device_map={"": c["device"]},
                                              attn_implementation=c["attention"]).eval()
    require(all(p.device.type == "cuda" for p in model.parameters()), "Unexpected CPU/offloaded generator parameters")
    require(not getattr(model.config, "quantization_config", None), "Quantization is not part of this protocol")
    require(model.get_input_embeddings().weight.dtype == getattr(torch, c["dtype"]), "Generator precision differs from the explicit config")
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    require(all(isinstance(i, int) for i in eos_ids), "Generator must define EOS token IDs")
    generation = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=c["max_new_tokens"],
                                  use_cache=True, eos_token_id=eos,
                                  pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_ids[0],
                                  bos_token_id=tokenizer.bos_token_id)
    check_frozen_runtime(meta, "generator", {**gpu, "effective_generation_config": generation.to_dict()})
    rows = read_jsonl(directory / "examples.jsonl")
    scores = read_jsonl(directory / "classifier_scores.jsonl")
    check_ids(scores, [r["id"] for r in rows])
    baseline = read_jsonl(directory / "baseline_predictions.jsonl")
    check_ids(baseline, [r["id"] for r in rows], complete=False)
    ensure_filtered(directory, rows, baseline, scores, settings, meta["identity"]["threshold"])
    elapsed = 0.0
    with torch.inference_mode():
        for i in range(len(baseline), len(rows)):
            row = rows[i]
            tokens, diagnostics = prompt_input(tokenizer, row["question"], row["context"], c)
            tensor = torch.tensor([tokens], device=c["device"], dtype=torch.long)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.generate(input_ids=tensor, attention_mask=torch.ones_like(tensor), generation_config=generation)
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            elapsed += duration
            new_tokens = output[0, len(tokens):].tolist()
            limit = len(new_tokens) >= c["max_new_tokens"] and (not new_tokens or new_tokens[-1] not in eos_ids)
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            prediction = {"id": row["id"], "mode": "baseline", **parse_output(raw, limit_reached=limit),
                          "raw_decoded_with_special_tokens": tokenizer.decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                          "generated_token_ids": new_tokens,
                          "generator_identity": digest(settings), "input_sha256": diagnostics["input_ids_sha256"],
                          "input": diagnostics, "new_tokens": len(new_tokens), "finish_reason": "length" if limit else "eos",
                          "generation_seconds": duration}
            if prediction["status"] != "ok":
                append_jsonl(directory / "execution_errors.jsonl", {
                    "utc": utc_now(), "attempt": meta["attempt"], "phase": "generator", "id": row["id"], "prediction": prediction})
                raise ValueError(f"Invalid generator output ({prediction['error']}) for {row['id']}; no complete result, raw output retained in execution_errors.jsonl")
            append_jsonl(directory / "baseline_predictions.jsonl", prediction)
            append_jsonl(directory / "filtered_predictions.jsonl",
                         gate_replay(prediction, scores[i]["p_answerable"], meta["identity"]["threshold"]))
            print(f"generator {i + 1}/{len(rows)} {row['id']} tokens={len(tokens)}+{len(new_tokens)} "
                  f"truncated={diagnostics['context_truncated']}", flush=True)
    write_json(directory / f"generator_phase_{meta['attempt']}.json", {
        **gpu, "effective_generation_config": generation.to_dict(), "completed_rows": len(rows),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "generation_seconds_this_attempt": elapsed, "filtered_timing": "not measured: paired replay"})


def launch_phase(phase, directory):
    """Distinct processes guarantee classifier and generator never share GPU residency."""
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "_worker", "--phase", phase, "--run", relative(directory)]
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    with (directory / "stdout.log").open("a", encoding="utf-8") as out, (directory / "stderr.log").open("a", encoding="utf-8") as err:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8", errors="replace")
        def pump(source, console, log):
            for line in source:
                log.write(line)
                log.flush()
                console.write(line)
                console.flush()
            source.close()
        threads = [threading.Thread(target=pump, args=(child.stdout, sys.stdout, out)),
                   threading.Thread(target=pump, args=(child.stderr, sys.stderr, err))]
        for thread in threads:
            thread.start()
        try:
            code = child.wait()
        except BaseException:
            child.terminate()
            child.wait()
            raise
        finally:
            for thread in threads:
                thread.join()
    require(code == 0, f"{phase} process failed (exit {code}); inspect stderr.log. No model, precision, offload or quantization fallback. OOM requires an explicit revised configuration and a new validation run.")


@contextlib.contextmanager
def run_lock(directory):
    path = directory / ".run.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "utc": utc_now()}, stream)
    except FileExistsError as exc:
        raise ValueError(f"Run is locked: {path}. Check its PID; remove only a stale lock before explicit resume.") from exc
    try:
        yield
    finally:
        path.unlink()


def records_for_run(directory):
    meta = read_json(directory / "run.json")
    rows = read_jsonl(directory / "examples.jsonl")
    b = read_jsonl(directory / "baseline_predictions.jsonl")
    f = read_jsonl(directory / "filtered_predictions.jsonl")
    scores = read_jsonl(directory / "classifier_scores.jsonl")
    ids = meta["identity"]["plan"]["ids"]
    check_ids(rows, ids)
    require(digest(rows) == meta["identity"]["plan"]["content_sha256"], "Saved evaluation examples changed")
    for records in (scores, b, f):
        check_ids(records, ids)
    for i, (before, after) in enumerate(zip(b, f)):
        require(before["status"] == "ok" and before["mode"] == "baseline", "Invalid baseline record")
        require(before["generator_identity"] == digest(meta["identity"]["settings"]), "Generator identity mismatch")
        require(scores[i]["classifier_identity"] == digest(meta["identity"]["settings"]["classifier"]), "Classifier score identity mismatch")
        require(after == gate_replay(before, scores[i]["p_answerable"], meta["identity"]["threshold"]), "Filtered replay mismatch")
    if meta["purpose"] == "test":
        require(len(ids) == 1000 and ids_checksum(ids) == meta["identity"]["settings"]["data"]["reference"]["splits"]["test"]["ids_sha256"],
                "Final run must cover exactly the fixed 1000 test IDs")
    return meta, rows, b, f


def save_analysis(directory, meta, rows, b, f, *, reviews=None, prefix=""):
    require(reviews is None or prefix.startswith("reviewed_"),
            "Supplementary review requires a reviewed_ prefix; primary automatic results must remain separate")
    c = meta["identity"]["settings"]["config"]
    report = paired_report(rows, b, f, reviews)
    report.update(split=meta["split"], purpose=meta["purpose"], technical_only=meta["purpose"] == "smoke")
    transitions = report.pop("transitions")
    write_json(directory / f"{prefix}metrics.json", report)
    write_json(directory / f"{prefix}paired_transitions.json", transitions)
    if meta["purpose"] != "smoke":
        bootstrap = cluster_bootstrap(rows, b, f, repeats=c["bootstrap_repeats"], seed=c["bootstrap_seed"], reviews=reviews)
        bootstrap.update(evaluation_split=meta["split"], purpose=meta["purpose"],
                         eligible_for_final_hypothesis_interpretation=meta["purpose"] == "test" and reviews is None)
        write_json(directory / f"{prefix}bootstrap.json", bootstrap)
    # IDs reference the immutable examples/predictions; do not duplicate their full texts.
    path = directory / f"{prefix}errors.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row, before, after in zip(rows, b, f):
            outcomes = {mode: score_row(row, prediction, (reviews or {}).get((mode, row["id"])))
                        for mode, prediction in (("baseline", before), ("filtered", after))}
            if any(value["outcome"] in ("incorrect_answer", "unsupported_answer", "unnecessary_refusal", "execution_error")
                   for value in outcomes.values()):
                stream.write(json.dumps({"id": row["id"], "context_group": row["context_group"], **outcomes}) + "\n")
    return report


def ensure_inputs(directory, rows, protocol):
    """Resume an interrupted reservation only before any inference output exists."""
    path = directory / "examples.jsonl"
    protocol_path = directory / "protocol.json"
    inference_started = any((directory / name).exists() for name in
                            ("classifier_scores.jsonl", "baseline_predictions.jsonl", "filtered_predictions.jsonl"))
    if not path.exists():
        require(not inference_started, "Inputs missing after inference; cannot reconstruct them over existing predictions")
        import tempfile

        fd, name = tempfile.mkstemp(prefix="examples.", suffix=".tmp", dir=directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    require(read_jsonl(path) == rows, "Saved inputs changed; cannot resume")
    if not protocol_path.exists():
        require(not inference_started, "Protocol missing after inference; cannot silently reconstruct it")
        write_json(protocol_path, protocol)
    require(read_json(protocol_path) == protocol, "Saved run protocol differs from its immutable plan")


def execute_run(args):
    purpose = args.command
    if purpose == "test":
        protocol_path = project_path(args.protocol)
        protocol = read_json(protocol_path)
        require(protocol["schema"] == "paired-llm-protocol-v1" and protocol["frozen"] is True, "A frozen protocol is required")
        unsigned = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
        require(digest(unsigned) == protocol["protocol_sha256"], "Protocol digest mismatch")
        require(protocol.get("evaluation_policy") == EVALUATION_POLICY,
                "Frozen protocol must explicitly specify primary automatic EM and optional supplementary review; no implicit policy upgrade")
        old = protocol["settings"]
        settings = descriptor(project_path(args.config), project_path(args.classifier_run),
                              project_path(args.classifier_model), project_path(args.generator_lock))
        check_resume(old, settings)
        require(settings["scoring"] == SCORING, "Scoring rules changed after protocol freeze")
        # This is the only command path allowed to open test.
        rows, signature = load_rows("test", settings["data"])
        plan = {"source": signature, "purpose": "test", "ids": [r["id"] for r in rows],
                "ids_sha256": ids_checksum(r["id"] for r in rows), "content_sha256": digest(rows)}
        threshold = protocol["threshold"]
        frozen = {"path": relative(protocol_path), "sha256": file_sha(protocol_path)}
    else:
        settings = descriptor(project_path(args.config), project_path(args.classifier_run),
                              project_path(args.classifier_model), project_path(args.generator_lock))
        rows, plan = validation_plan(settings, purpose)
        threshold, frozen = .5, None
    require(re.fullmatch(r"\d{8}T\d{6}\d{0,6}Z_[A-Za-z0-9_-]+", args.run_id), "Use a UTC timestamp followed by a descriptive run_id")
    directory = OUTPUTS / args.run_id
    identity = {"settings": settings, "plan": plan, "threshold": threshold, "protocol": frozen}
    if args.resume:
        require(directory.is_dir(), "--resume requires an existing run")
        meta = read_json(directory / "run.json")
        check_resume(meta["identity"], identity)
        require(meta["purpose"] == purpose, "Resume purpose changed")
        require(meta["status"] != "completed", "Run already completed; inspect its reports")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        meta = {"schema": "paired-llm-run-v1", "run_id": args.run_id, "status": "starting",
                "purpose": purpose, "split": "test" if purpose == "test" else "validation",
                "technical_only": purpose == "smoke", "identity": identity, "attempt": 0,
                "created_at_utc": utc_now(), "duration_seconds": 0.0,
                "timing_note": "baseline generation measured; filtered uses paired replay, no gated-inference speedup measured"}
        write_json(directory / "run.json", meta)
    planned_protocol = protocol if purpose == "test" else {
        "schema": "development-plan-v1", "frozen": False, "settings": settings, "threshold": threshold, "subset": plan}
    started = time.perf_counter()
    with run_lock(directory):
        try:
            ensure_inputs(directory, rows, planned_protocol)
            for name in ("classifier_scores.jsonl", "baseline_predictions.jsonl", "filtered_predictions.jsonl"):
                records = read_jsonl(directory / name, recover=args.resume)
                check_ids(records, plan["ids"], complete=False)
            meta.update(status="running", attempt=meta["attempt"] + 1, last_started_at_utc=utc_now())
            meta.pop("error", None)
            write_json(directory / "run.json", meta, replace=True)
            scores = read_jsonl(directory / "classifier_scores.jsonl")
            if len(scores) != len(rows):
                meta["phase"] = "classifier"
                write_json(directory / "run.json", meta, replace=True)
                launch_phase("classifier", directory)
            # The classifier worker has exited and released all GPU allocations.
            meta["phase"] = "generator"
            write_json(directory / "run.json", meta, replace=True)
            launch_phase("generator", directory)
            checked, examples, before, after = records_for_run(directory)
            # Preserve any previous partial analysis; finish atomically from complete paired predictions.
            for name in ("metrics.json", "paired_transitions.json", "bootstrap.json", "errors.jsonl"):
                if (directory / name).exists():
                    require(args.resume, "Analysis already exists without explicit resume")
                    (directory / name).rename(directory / f"attempt_{meta['attempt'] - 1}_{name}")
            save_analysis(directory, checked, examples, before, after)
            meta.update(status="completed", phase="completed", completed_at_utc=utc_now(),
                        completed_unique_ids=len(rows), errors_unresolved=0,
                        artifacts_sha256={p.name: file_sha(p) for p in directory.iterdir()
                                          if p.is_file() and (p.name.startswith(("classifier_phase_", "generator_phase_")) or p.name in ("examples.jsonl", "classifier_scores.jsonl", "baseline_predictions.jsonl",
                                                                       "filtered_predictions.jsonl", "protocol.json", "metrics.json", "paired_transitions.json", "bootstrap.json", "errors.jsonl"))})
        except BaseException as exc:
            meta.update(status="failed", failed_at_utc=utc_now(), errors_unresolved=True, error=f"{type(exc).__name__}: {exc}")
            append_jsonl(directory / "execution_errors.jsonl", {"utc": utc_now(), "attempt": meta["attempt"],
                         "phase": meta.get("phase"), "error": meta["error"], "traceback": traceback.format_exc()})
            raise
        finally:
            meta["duration_seconds"] += time.perf_counter() - started
            write_json(directory / "run.json", meta, replace=True)
    print(f"COMPLETED {purpose}: {len(rows)} paired examples\n{directory}")
    show(directory)


def verify_completed(directory, purpose=None):
    meta = read_json(directory / "run.json")
    require(meta["status"] == "completed" and (purpose is None or meta["purpose"] == purpose), "Requires a completed run of the requested purpose")
    for name, expected in meta["artifacts_sha256"].items():
        require(file_sha(directory / name) == expected, f"Saved artifact changed: {name}")
    return records_for_run(directory)


def freeze_protocol(args):
    directory, smoke_dir = project_path(args.development_run), project_path(args.smoke_run)
    meta, rows, b, f = verify_completed(directory, "development")
    smoke, _, _, _ = verify_completed(smoke_dir, "smoke")
    settings = descriptor(project_path(args.config), project_path(args.classifier_run),
                          project_path(args.classifier_model), project_path(args.generator_lock))
    check_resume(settings, meta["identity"]["settings"])
    check_resume(settings, smoke["identity"]["settings"])
    _, current_plan = validation_plan(settings, "development")
    check_resume(current_plan, meta["identity"]["plan"])
    runtime = {}
    for phase in ("classifier", "generator"):
        fields = ("gpu", "total_bytes", "cuda_build", "precision", "deterministic_algorithms", "tf32", "inference_seed",
                  "tokenizer" if phase == "classifier" else "effective_generation_config")
        actual, smoke_actual = latest_phase(directory, phase), latest_phase(smoke_dir, phase)
        runtime[phase + "_runtime"] = {key: actual[key] for key in fields}
        require(runtime[phase + "_runtime"] == {key: smoke_actual[key] for key in fields},
                f"Smoke and development actual {phase} settings differ; validate the same configuration")
    threshold = args.threshold
    require(math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid threshold")
    threshold_rule = {"rule": "Pre-specified threshold 0.5; no test tuning"}
    if threshold != .5:
        require(args.threshold_report, "Non-default threshold requires an explicit validation threshold report")
        report_path = project_path(args.threshold_report)
        report = read_json(report_path)
        require(report["split"] == "validation" and report["development_run_sha256"] == file_sha(directory / "run.json")
                and threshold in [r["threshold"] for r in report["results"]], "Threshold report does not match this validation run/threshold")
        threshold_rule = {"rule": "Explicit manual choice from recorded validation-only sensitivity study; not selected on test",
                          "report_path": relative(report_path), "sha256": file_sha(report_path)}
    frozen = {"schema": "paired-llm-protocol-v1", "frozen": True, "frozen_at_utc": utc_now(),
              "evaluation_policy": EVALUATION_POLICY,
              "settings": settings, **runtime, "threshold": threshold, "threshold_selection": threshold_rule,
              "validation_subset": meta["identity"]["plan"],
              "development_evidence": {"path": relative(directory), "run_sha256": file_sha(directory / "run.json"),
                                       "metrics_sha256": file_sha(directory / "metrics.json"),
                                       "scores_sha256": file_sha(directory / "classifier_scores.jsonl")},
              "smoke_evidence": {"path": relative(smoke_dir), "run_sha256": file_sha(smoke_dir / "run.json")},
              "metrics": list(paired_report(rows, b, f)["filtered_minus_baseline_pp"]),
              "hypotheses": SCORING["hypotheses"], "review_rubric": SCORING["review_rubric"],
              "test_rule": "Exactly all 1000 fixed ordered manifest IDs, no subset/limit/debug mode; no settings changed after freeze"}
    frozen["protocol_sha256"] = digest(frozen)
    write_json(project_path(args.output), frozen)
    print(f"Protocol frozen: {project_path(args.output)}\nNo test examples loaded.")


def threshold_study(args):
    directory = project_path(args.run)
    meta, rows, baseline, _ = verify_completed(directory, "development")
    scores = read_jsonl(directory / "classifier_scores.jsonl")
    results = []
    for threshold in args.thresholds:
        require(math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid threshold")
        f = [gate_replay(b, s["p_answerable"], threshold) for b, s in zip(baseline, scores)]
        results.append({"threshold": threshold, "metrics": paired_report(rows, baseline, f)})
    write_json(project_path(args.output), {"split": "validation", "purpose": "explicit sensitivity study; does not change default threshold",
                                         "development_run_sha256": file_sha(directory / "run.json"), "results": results})
    print(f"Validation-only threshold study: {project_path(args.output)}")


def show(directory, index=None):
    meta = read_json(directory / "run.json")
    print(f"{meta['run_id']}: status={meta['status']}, purpose={meta['purpose']}, split={meta['split']}")
    if meta["purpose"] == "smoke":
        print("TECHNICAL SMOKE ONLY: excluded from hypothesis evaluation and final statistics")
    if meta["status"] != "completed":
        print("Incomplete run: no completed evaluation claimed. Inspect execution_errors.jsonl and stderr.log.")
        return
    _, rows, b, f = verify_completed(directory)
    report = read_json(directory / "metrics.json")
    print("PRIMARY SCORING: SQuAD normalized exact match with any gold answer; existing refusal/unanswerable rules.")
    print("Token F1 is diagnostic. Manual review is optional supplementary analysis and does not replace automatic H1/H2 outcomes.")
    print("LIMITATION: EM can reject semantically correct paraphrases.")
    if meta["purpose"] != "test":
        print("DEVELOPMENT ONLY: these validation results do not confirm H1/H2.")
    for mode in ("baseline", "filtered"):
        print(f"\n{mode}: {json.dumps(report[mode]['counts'])}")
        for key, value in report[mode]["rates"].items():
            print(f"{key}: {value['count']}/{value['denominator']} = {value['rate']}")
    print("filtered - baseline (pp):", json.dumps(report["filtered_minus_baseline_pp"]))
    print(f"Prevented unsupported: {report['prevented_unsupported_answers']}; lost correct: {report['lost_correct_answers']}")
    print(report["timing_note"])
    if (directory / "bootstrap.json").is_file():
        print("Paired cluster bootstrap:", json.dumps(read_json(directory / "bootstrap.json"), indent=2))
    if index is not None:
        require(0 <= index < len(rows), "Example index outside this run")
        print(json.dumps({"example": rows[index], "baseline": b[index], "filtered": f[index],
                          "automatic_scores": {"baseline": score_row(rows[index], b[index]),
                                               "filtered": score_row(rows[index], f[index])}}, indent=2, ensure_ascii=False))


def review_source(directory):
    return {"run": relative(directory), "run_sha256": file_sha(directory / "run.json"),
            "files_sha256": {name: file_sha(directory / name) for name in (
                "protocol.json", "examples.jsonl", "baseline_predictions.jsonl", "filtered_predictions.jsonl")}}


def saved_review_rubric(directory, meta):
    rubric = read_json(directory / "protocol.json")["settings"]["scoring"]["review_rubric"]
    require(rubric == meta["identity"]["settings"]["scoring"]["review_rubric"], "Run and protocol review rubrics differ")
    return rubric


def review_export(args):
    directory = project_path(args.run)
    meta, rows, b, f = verify_completed(directory)
    require(meta["identity"]["settings"]["scoring"] == SCORING, "Scoring/review rubric changed after this run")
    destination = project_path(args.output)
    require(not destination.exists(), "Review destination exists")
    candidates = review_candidates(rows, b, f)
    rubric = saved_review_rubric(directory, meta)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "html":
        require(destination.suffix.lower() == ".html", "HTML assessment output must use .html")
        mapping_path = destination.with_suffix(".mapping.json")
        require(not mapping_path.exists(), "Assessment mapping destination exists")
        form, mapping = review_bundle(candidates, rubric, review_source(directory))
        html = assessment_html(form)
        write_json(mapping_path, mapping)
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(html)
        print(f"UNASSESSED export: {len(form['cards'])} unique ID/answer cards from {len(candidates)} mode/ID candidates. "
              "Decisions and rationales are blank; EM matches are included.\n"
              f"Form (mode membership hidden): {destination}\nMapping: {mapping_path}\n"
              "Fill the form manually, save draft JSON to continue later, then export completed decisions. No reviewed metrics produced.")
        return
    with destination.open("x", encoding="utf-8") as stream:
        for row in candidates:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Review rubric: {json.dumps(rubric, indent=2)}\nUnassessed review rows: {len(candidates)}\n{destination}")


def review_import(args):
    directory = project_path(args.run)
    meta, rows, b, f = verify_completed(directory)
    require(meta["identity"]["settings"]["scoring"] == SCORING, "Scoring/review rubric changed after this run")
    decisions_path = project_path(args.decisions)
    candidates = review_candidates(rows, b, f)
    rubric = saved_review_rubric(directory, meta)
    mapping_path = project_path(args.mapping) if args.mapping else None
    if mapping_path:
        decision_document = read_json(decisions_path)
        reviews, decisions = expand_review_bundle(candidates, rubric, review_source(directory),
                                                 read_json(mapping_path), decision_document)
    else:
        decisions = read_jsonl(decisions_path)
        reviews = validate_reviews(candidates, decisions)
    destination = project_path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    save_analysis(destination, meta, rows, b, f, reviews=reviews, prefix="reviewed_")
    write_json(destination / "review.json", {"source_run": relative(directory), "run_sha256": file_sha(directory / "run.json"),
                                           "decisions_sha256": file_sha(decisions_path), "decisions": decisions,
                                           "rubric": rubric, "mapping_sha256": file_sha(mapping_path) if mapping_path else None,
                                           "raw_predictions_modified": False})
    print(f"Reviewed results (separate from automatic EM): {destination}")
    print(json.dumps(read_json(destination / "reviewed_metrics.json"), indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    commands.add_parser("classifiers", help="List eligible completed full classifier runs and the selection rule")
    prep = commands.add_parser("prepare-generator", help="Explicit network download and immutable model/tokenizer lock; no inference")
    prep.add_argument("--config", default=relative(CONFIG))
    prep.add_argument("--revision", default="main", help="Resolve this ref to a concrete commit SHA before downloading")
    prep.add_argument("--output", required=True, help="New generator lock JSON inside the project")
    for name in ("smoke", "development", "test", "freeze"):
        command = commands.add_parser(name, help={"smoke": "Technical fixed validation subset; GPU inference",
            "development": "Fixed validation subset; GPU inference", "test": "Explicit final evaluation of all 1000 test examples",
            "freeze": "Freeze validated settings and rubric; never open test"}[name])
        command.add_argument("--config", default=relative(CONFIG))
        command.add_argument("--classifier-run", required=True)
        command.add_argument("--classifier-model", required=True)
        command.add_argument("--generator-lock", required=True)
        if name == "freeze":
            command.add_argument("--smoke-run", required=True)
            command.add_argument("--development-run", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--threshold", type=float, default=.5)
            command.add_argument("--threshold-report")
        else:
            command.add_argument("--local-files-only", action="store_true", required=True, help="Inference is always offline; this explicit flag is required")
            command.add_argument("--run-id", required=True)
            command.add_argument("--resume", action="store_true", help="Explicitly resume missing IDs after checking identity; no implicit overwrite")
        if name == "test":
            command.add_argument("--protocol", required=True)
    command = commands.add_parser("thresholds", help="Explicit validation-only threshold sensitivity analysis")
    command.add_argument("--run", required=True)
    command.add_argument("--thresholds", type=float, nargs="+", required=True)
    command.add_argument("--output", required=True)
    for name in ("show", "example", "review-export", "review-import"):
        command = commands.add_parser(name)
        command.add_argument("--run", required=True)
        if name == "example":
            command.add_argument("--index", type=int, default=0)
        elif name == "review-export":
            command.add_argument("--output", required=True)
            command.add_argument("--format", choices=("jsonl", "html"), default="jsonl",
                                 help="HTML deduplicates identical ID/answers, hides mode membership and writes a separate mapping")
        elif name == "review-import":
            command.add_argument("--decisions", required=True)
            command.add_argument("--output-dir", required=True)
            command.add_argument("--mapping", help="Mapping sidecar for completed HTML assessment JSON; omit for legacy mode/ID JSONL")
    command = commands.add_parser("_worker", help="Internal captured process; do not invoke manually")
    command.add_argument("--phase", choices=("classifier", "generator"), required=True)
    command.add_argument("--run", required=True)
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == "classifiers":
            list_classifiers()
        elif args.command == "prepare-generator":
            prepare_generator(args)
        elif args.command in ("smoke", "development", "test"):
            execute_run(args)
        elif args.command == "freeze":
            freeze_protocol(args)
        elif args.command == "thresholds":
            threshold_study(args)
        elif args.command in ("show", "example"):
            show(project_path(args.run), getattr(args, "index", None))
        elif args.command == "review-export":
            review_export(args)
        elif args.command == "review-import":
            review_import(args)
        elif args.command == "_worker":
            configure_cache()
            directory = project_path(args.run)
            meta = read_json(directory / "run.json")
            require(meta["status"] == "running" and meta["phase"] == args.phase
                    and (directory / ".run.lock").exists(), "Worker requires a reserved running experiment and matching phase")
            (worker_classifier if args.phase == "classifier" else worker_generator)(directory, meta)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if args.command == "_worker":
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
