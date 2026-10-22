"""Train and validate an answerability classifier; never open the held-out test split."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from report_results import DEV, INITIAL, metric_values, show_report

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "squad_v2_answerability"
OUTPUTS = ROOT / "outputs" / "classifier"


def initialize_ml(local_files_only=False):
    # Import ML only in the captured worker, so all library messages reach stderr.log.
    os.environ.update(HF_HOME=str(ROOT / ".cache" / "huggingface"),
                      HF_HUB_CACHE=str(ROOT / ".cache" / "huggingface" / "hub"),
                      HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false")
    if local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    global np, torch, Dataset, concatenate_datasets, disable_progress_bars, load_from_disk
    global snapshot_download, AutoModelForSequenceClassification, AutoTokenizer
    global DataCollatorWithPadding, Trainer, TrainerCallback, TrainingArguments, set_seed
    global accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    import numpy as np
    import torch
    from datasets import Dataset, concatenate_datasets, disable_progress_bars, load_from_disk
    from huggingface_hub import snapshot_download
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def data_files_sha256():
    result = {}
    # Deliberately inspect only the two permitted directories, never test.
    for name in ("train", "validation"):
        for path in sorted((PROCESSED / name).rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    result[path.relative_to(PROCESSED).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def gpu_probe():
    require(torch.cuda.is_available(), "CUDA is unavailable in this PyTorch environment; full CPU training is prohibited.")
    properties = torch.cuda.get_device_properties(0)
    result = {
        "torch_version": torch.__version__, "torch_version_cuda": torch.version.cuda,
        "cuda_is_available": True, "name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "memory_total_bytes": properties.total_memory,
        "supported_architectures": torch.cuda.get_arch_list(),
    }
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    result["nvidia_smi"] = {"exit_code": smi.returncode, "stdout": smi.stdout, "stderr": smi.stderr}
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    (x @ x.T).square().mean().backward()
    torch.cuda.synchronize()
    require(bool(torch.isfinite(x.grad).all().item()), "Float32 CUDA backward produced nonfinite gradients.")
    result["float32_forward_backward_passed"] = True
    result["precision_probes"] = []
    modes = ["bf16", "fp16", "fp32"] if torch.cuda.is_bf16_supported() else ["fp16", "fp32"]
    for mode in modes:
        try:
            x = torch.randn(64, 64, device="cuda", requires_grad=True)
            scaler = torch.amp.GradScaler("cuda", enabled=mode == "fp16")
            dtype = torch.bfloat16 if mode == "bf16" else torch.float16
            autocast = torch.autocast("cuda", dtype=dtype) if mode != "fp32" else contextlib.nullcontext()
            with autocast:
                loss = ((x @ x.T) / 64).square().mean()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            require(bool(torch.isfinite(x.grad).all().item()), f"Nonfinite {mode} gradients.")
            result["precision_probes"].append({"mode": mode, "passed": True})
            result["precision"] = mode
            break
        except Exception as exc:
            result["precision_probes"].append({"mode": mode, "passed": False, "error": str(exc)})
    require("precision" in result, "All GPU precision probes failed.")
    del x
    torch.cuda.empty_cache()
    return result


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float64)
    exponentials = np.exp(values - values.max(axis=1, keepdims=True))
    return exponentials[:, 1] / exponentials.sum(axis=1)


def metrics(labels, probability):
    predicted = (np.asarray(probability) >= 0.5).astype(np.int64)
    report = classification_report(labels, predicted, labels=[0, 1], output_dict=True, zero_division=0)
    return {
        "threshold": 0.5, "class_order": [0, 1],
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "per_class": {name: report[name] for name in ("0", "1")},
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_axes": "rows=true, columns=predicted; order [0=unanswerable, 1=answerable]",
    }


def tokenize_and_diagnose(source, tokenizer, config, name):
    counts = {
        "rows": len(source), "context_truncated": 0, "answerable": 0,
        "truncated_answerable": 0, "answerable_no_fully_visible_gold_span": 0,
        "truncated_answerable_no_fully_visible_gold_span": 0,
        "invalid_gold_spans": 0, "max_original_context_tokens": 0,
    }
    pieces = []
    for start in range(0, len(source), 256):
        rows = source[start:start + 256]
        encoding = tokenizer(
            rows["question"], rows["context"], max_length=config["max_length"],
            truncation="only_second", padding=False, return_offsets_mapping=True,
            return_token_type_ids=False,
        )
        full_contexts = tokenizer(rows["context"], add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        for i, original_context_ids in enumerate(full_contexts):
            offsets = [pair for pair, segment in zip(encoding["offset_mapping"][i], encoding.sequence_ids(i))
                       if segment == 1 and pair[1] > pair[0]]
            truncated = len(offsets) < len(original_context_ids)
            counts["context_truncated"] += int(truncated)
            counts["max_original_context_tokens"] = max(counts["max_original_context_tokens"], len(original_context_ids))
            if rows["label"][i] == 1:
                counts["answerable"] += 1
                counts["truncated_answerable"] += int(truncated)
                visible_start = min((pair[0] for pair in offsets), default=0)
                visible_end = max((pair[1] for pair in offsets), default=0)
                visible = False
                answers = rows["answers"][i]
                require(len(answers["text"]) == len(answers["answer_start"]), "Inconsistent gold span metadata.")
                for answer, answer_start in zip(answers["text"], answers["answer_start"]):
                    end = answer_start + len(answer)
                    valid = 0 <= answer_start < end <= len(rows["context"][i]) and rows["context"][i][answer_start:end] == answer
                    counts["invalid_gold_spans"] += int(not valid)
                    visible |= valid and bool(offsets) and visible_start <= answer_start and end <= visible_end
                counts["answerable_no_fully_visible_gold_span"] += int(not visible)
                counts["truncated_answerable_no_fully_visible_gold_span"] += int(truncated and not visible)
        model_inputs = {key: encoding[key] for key in ("input_ids", "attention_mask")}
        model_inputs["labels"] = rows["label"]
        require(set(model_inputs) == {"input_ids", "attention_mask", "labels"}, "Unexpected model input columns.")
        pieces.append(Dataset.from_dict(model_inputs))
        if start % 10240 == 0:
            print(f"Tokenizing/diagnosing {name}: {min(start + 256, len(source))}/{len(source)}", flush=True)
    counts["context_truncated_fraction"] = counts["context_truncated"] / counts["rows"]
    counts["answerable_no_fully_visible_gold_span_fraction"] = counts["answerable_no_fully_visible_gold_span"] / counts["answerable"]
    counts["truncated_answerable_no_fully_visible_gold_span_fraction"] = counts["truncated_answerable_no_fully_visible_gold_span"] / max(1, counts["truncated_answerable"])
    counts["diagnostic_only"] = "Offsets/answers are excluded from model inputs; prefix context truncation, one sequence per example, labels unchanged."
    return concatenate_datasets(pieces), counts


def fresh_model(snapshot, seed):
    seed_everything(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, num_labels=2,
        problem_type="single_label_classification",
        id2label={0: "unanswerable", 1: "answerable"},
        label2id={"unanswerable": 0, "answerable": 1},
    )
    # DistilBERT's CE is a per-microbatch mean; its **kwargs do not implement
    # num_items_in_batch. Prevent Trainer from treating it as GA-normalized loss.
    model.accepts_loss_kwargs = False
    return model


def verify_accumulation(snapshot, tokenizer, train, seed):
    model = fresh_model(snapshot, seed).eval().to("cuda")
    collator = DataCollatorWithPadding(tokenizer)
    batch = {key: value.to("cuda") for key, value in collator([train[index] for index in range(4)]).items()}
    output = model(**batch)
    reference_ce = torch.nn.functional.cross_entropy(output.logits, batch["labels"])
    require(torch.allclose(output.loss, reference_ce), "Loss is not standard unweighted cross-entropy.")
    weight = model.classifier.weight
    reference_gradient = torch.autograd.grad(output.loss, weight)[0]
    accumulated_gradient = torch.zeros_like(reference_gradient)
    micro_loss = 0.0
    for start in (0, 2):
        micro = {key: value[start:start + 2] for key, value in batch.items()}
        loss = model(**micro).loss / 2
        micro_loss += float(loss.detach())
        accumulated_gradient += torch.autograd.grad(loss, weight)[0]
    difference = float((reference_gradient - accumulated_gradient).abs().max())
    require(torch.allclose(reference_gradient, accumulated_gradient, atol=1e-5, rtol=1e-3),
            "Microbatch accumulation differs from the full-batch mean CE gradient.")
    result = {"passed": True, "split": "train", "rows": 4, "mode": "float32 eval (dropout disabled) with autograd",
              "loss": float(output.loss.detach()), "mean_microbatch_loss": micro_loss,
              "max_abs_classifier_gradient_difference": difference, "atol": 1e-5, "rtol": 1e-3,
              "trainer_model_accepts_loss_kwargs": False,
              "normalization": "Per-microbatch mean CE divided by current gradient accumulation steps by Trainer."}
    del model, output, batch, reference_gradient, accumulated_gradient, weight
    gc.collect()
    torch.cuda.empty_cache()
    return result


def prediction_probe(model, tokenizer, raw_validation):
    inputs = tokenizer(raw_validation[:8]["question"], raw_validation[:8]["context"],
                       max_length=512, truncation="only_second", padding=True,
                       return_token_type_ids=False, return_tensors="pt")
    model.eval().to("cuda")
    # Accelerate can leave an autocast wrapper on the trained model.forward.
    # Call the original forward so both sides of the disk check use real float32.
    forward = getattr(model, "_original_forward", model.forward)
    with torch.inference_mode(), torch.autocast("cuda", enabled=False):
        return forward(**{key: value.to("cuda") for key, value in inputs.items()}).logits.float().cpu().numpy()


def model_checksum(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])


def ordered_checksum(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_checksums(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(directory).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


class MetricComputer:
    def __init__(self):
        self.last = None

    def __call__(self, output):
        self.last = metrics(output.label_ids, probabilities(output.predictions))
        return metric_values(self.last)


def balanced_subset(source, per_class, seed):
    rng = random.Random(seed)
    labels = list(source["label"])
    indices = []
    for label in (0, 1):
        candidates = [i for i, value in enumerate(labels) if value == label]
        indices.extend(rng.sample(candidates, per_class))
    return source.select(sorted(indices), keep_in_memory=True)


@contextlib.contextmanager
def training_lock(run_dir):
    path = ROOT / ".cache" / "classifier_training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError(f"Training lock exists: {path}. Check the recorded PID before removing a stale lock.") from exc
    try:
        with handle:
            json.dump({"pid": os.getpid(), "run_directory": str(run_dir)}, handle)
        yield
    finally:
        path.unlink()


def metric_report(model_metrics, config, smoke, rows, stage, baseline=None):
    report = {"evaluation_label": DEV, "evaluation_split": "validation", "test_evaluated": False,
              "training_seed": config["seed"], "split_seed": 42, "smoke": smoke, "rows": rows,
              "stage": stage, "model": model_metrics}
    if baseline is not None:
        report["always_answerable"] = baseline
    return report


def callbacks(computer, run_dir, config, smoke, rows):
    class History(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            event = {"utc": utc_now(), "step": state.global_step, "epoch": state.epoch, **(logs or {})}
            with (run_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            epoch = round(state.epoch)
            require(abs(state.epoch - epoch) < 1e-6, "Evaluation must occur at an epoch boundary.")
            report = metric_report(computer.last, config, smoke, rows, f"epoch_{epoch}")
            report.update(epoch=epoch, optimizer_steps=state.global_step)
            with (run_dir / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            show_report(report, f"EPOCH {epoch} VALIDATION")
    return [History()]


def train_attempt(run_dir, snapshot, tokenizer, train, validation, raw_validation,
                  config, info, smoke, batch_size, attempt, initial_hash):
    """One training loop for both full runs and small smoke subsets."""
    accumulation = 16 // batch_size
    computer = MetricComputer()
    model = trainer = None
    before = initial_hash
    try:
        model = fresh_model(snapshot, config["seed"])
        arguments = TrainingArguments(
            output_dir=str(run_dir / "checkpoints" / f"attempt_{attempt}"),
            num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"], warmup_steps=config["warmup_ratio"],
            per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
            per_device_eval_batch_size=min(config["eval_batch_size"], batch_size),
            dataloader_num_workers=0, seed=config["seed"], data_seed=config["seed"],
            eval_strategy="epoch", save_strategy="epoch", logging_steps=1 if smoke else config["logging_steps"],
            load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
            save_total_limit=2, report_to=[], disable_tqdm=True, bf16=info["gpu"]["precision"] == "bf16",
            fp16=info["gpu"]["precision"] == "fp16", tf32=False, optim="adamw_torch",
            lr_scheduler_type="linear", max_grad_norm=1.0, label_smoothing_factor=0.0,
            dataloader_pin_memory=True,
        )
        info["training_arguments"] = arguments.to_dict()
        info["phase"] = "initial_validation"
        write_json(run_dir / "run.json", info)
        trainer = Trainer(model=model, args=arguments, train_dataset=train, eval_dataset=validation,
                          processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                          compute_metrics=computer, callbacks=callbacks(computer, run_dir, config, smoke, len(validation)))
        require(trainer.model_accepts_loss_kwargs is False, "Incorrect gradient accumulation normalization.")
        seed_everything(config["seed"])
        rng = capture_rng()
        before = model_checksum(model)
        require(initial_hash is None or initial_hash == before, "OOM restart changed seeded initialization.")
        print("\nINITIAL VALIDATION BEFORE THE FIRST OPTIMIZER STEP", flush=True)
        print(INITIAL, flush=True)
        output = trainer.predict(validation, metric_key_prefix="initial_validation")
        require(np.array_equal(output.label_ids, np.asarray(raw_validation["label"])), "Initial validation order differs.")
        after = model_checksum(model)
        require(before == after and trainer.state.global_step == 0, "Initial evaluation updated weights or optimizer steps.")
        require(all(parameter.grad is None for parameter in model.parameters()), "Initial evaluation accumulated gradients.")
        restore_rng(rng)
        info["initial_weights_check"] = {"before_sha256": before, "after_sha256": after,
                                          "unchanged": True, "optimizer_steps": 0, "rng_restored": True,
                                          "same_model_used_for_training": True}
        initial = metric_report(computer.last, config, smoke, len(validation), "initial")
        initial["description"] = INITIAL
        write_json(run_dir / "initial_validation_metrics.json", initial)
        show_report(initial, "INITIAL VALIDATION")
        del output
        seed_everything(config["seed"])
        require(model_checksum(model) == before, "Weights changed before training.")
        info["phase"] = "training"
        write_json(run_dir / "run.json", info)
        print(f"\nTRAINING: seed={config['seed']} batch={batch_size} accumulation={accumulation} effective_batch=16 precision={info['gpu']['precision']}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trained = trainer.train()
        torch.cuda.synchronize()
        require(trainer.state.global_step > 0 and trainer.state.epoch >= config["epochs"] - 1e-6,
                "Requested training did not complete.")
        for row in trainer.state.log_history:
            for key in ("loss", "grad_norm", "eval_loss"):
                require(key not in row or math.isfinite(float(row[key])), f"Nonfinite {key}.")
        trainer.save_state()
        summary = {"completed_epochs": trainer.state.epoch, "global_steps": trainer.state.global_step,
                   "duration_seconds": time.perf_counter() - started, "train_batch_size": batch_size,
                   "gradient_accumulation_steps": accumulation, "eval_batch_size": min(config["eval_batch_size"], batch_size),
                   "effective_batch_size": 16, "precision": info["gpu"]["precision"],
                   "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "best_checkpoint": trainer.state.best_model_checkpoint, "best_validation_macro_f1": trainer.state.best_metric,
                   "criterion": "validation macro-F1", "metrics": trained.metrics,
                   "rng_protocol": "Restore RNG after initial evaluation, then explicitly reseed before Trainer.train."}
        return trainer, summary, before
    except torch.cuda.OutOfMemoryError as exc:
        record = {"attempt": attempt, "status": "discarded_oom", "error": str(exc),
                  "train_batch_size": batch_size, "gradient_accumulation_steps": accumulation,
                  "eval_batch_size": min(config["eval_batch_size"], batch_size), "effective_batch_size": 16}
        failed_dir = run_dir / "attempts" / f"attempt_{attempt}"
        failed_dir.mkdir(parents=True, exist_ok=False)
        # Preserve evidence of an interrupted attempt before starting new files.
        for name in ("initial_validation_metrics.json", "epoch_metrics.jsonl", "history.jsonl"):
            if (run_dir / name).exists():
                (run_dir / name).rename(failed_dir / name)
        write_json(failed_dir / "failure.json", {**record, "training_arguments": info["training_arguments"]})
        exc.__traceback__ = None
        trainer = model = None
        gc.collect()
        torch.cuda.empty_cache()
        return None, record, before


def run_experiment(run_dir, info, smoke):
    started = time.perf_counter()
    config = info["config"]
    model_dir = Path(info["model_directory"])
    disable_progress_bars()
    torch.set_num_threads(8)
    info.update(status="running", gpu=gpu_probe())
    info["versions"] = {name: version(name) for name in ("torch", "transformers", "accelerate", "scikit-learn", "datasets", "numpy", "tokenizers", "huggingface-hub")}
    info["versions"]["python"] = platform.python_version()
    code_dir = run_dir / "code"
    code_dir.mkdir()
    for name in ("train_classifier.py", "report_results.py"):
        shutil.copyfile(ROOT / "scripts" / name, code_dir / name)
    shutil.copyfile(ROOT / "requirements.txt", code_dir / "requirements.txt")
    shutil.copyfile(ROOT / "configs" / "data_split.json", code_dir / "data_split.json")
    info["source_code_sha256"] = file_checksums(code_dir)
    snapshot = snapshot_download(config["model_id"], revision=config["revision"], local_files_only=info["local_files_only"],
                                 cache_dir=str(ROOT / ".cache" / "huggingface" / "hub"),
                                 allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"])
    info["model_source"] = {"model_id": config["model_id"], "revision": config["revision"],
                            "tokenizer_revision": config["revision"], "snapshot_path": snapshot}
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
    require(tokenizer.is_fast, "Fast tokenizer is required for offset diagnostics.")
    source_hashes = data_files_sha256()
    raw_train = load_from_disk(str(PROCESSED / "train"))
    raw_validation = load_from_disk(str(PROCESSED / "validation"))
    split_report = read_json(ROOT / "outputs" / "split_summary.json")
    reference_path = ROOT / "configs" / "data_split.json"
    reference = read_json(reference_path)
    require(reference["settings"]["seed"] == 42 and split_report["checks"].get("matches_reference_config"),
            "Expected splits verified against configs/data_split.json with split seed=42. Run prepare_splits.py first.")
    require(split_report["identity"]["reference_sha256"] == hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "Split reference changed after preparation; recheck saved splits before training.")
    for name, source in (("train", raw_train), ("validation", raw_validation)):
        require(len(source) == reference["splits"][name]["rows"], f"Unexpected {name} size versus configs/data_split.json.")
        require(ordered_checksum(source["id"]) == reference["splits"][name]["ids_sha256"], f"Unexpected {name} IDs/order versus configs/data_split.json.")
        require(all(int(bool(answer["text"])) == label for answer, label in zip(source["answers"], source["label"])), f"Invalid {name} labels.")
    source_rows = {"train": len(raw_train), "validation": len(raw_validation)}
    if smoke:
        raw_train = balanced_subset(raw_train, 16, 42)
        raw_validation = balanced_subset(raw_validation, 16, 42)
        print("SMOKE ONLY: 32 train / 32 validation rows; same training loop; excluded from full-run comparisons.", flush=True)
    info["data"] = {"files_sha256": source_hashes, "source_rows": source_rows,
                    "rows": {"train": len(raw_train), "validation": len(raw_validation)}, "split_seed": 42,
                    "ordered_ids_sha256": {"train": ordered_checksum(raw_train["id"]), "validation": ordered_checksum(raw_validation["id"])},
                    "model_revision": config["revision"], "max_length": config["max_length"],
                    "input_fields": ["question", "context"], "truncation": "only_second",
                    "label_only_for_supervision": True, "test_accessed": False}
    train, train_diagnostics = tokenize_and_diagnose(raw_train, tokenizer, config, "train")
    validation, validation_diagnostics = tokenize_and_diagnose(raw_validation, tokenizer, config, "validation")
    info["tokenization"] = {"train": train_diagnostics, "validation": validation_diagnostics}
    info["gradient_accumulation_check"] = verify_accumulation(snapshot, tokenizer, train, config["seed"])
    write_json(run_dir / "run.json", info)
    batch_size, attempt, initial_hash = config["train_batch_size"], 0, None
    info["oom_restarts"] = []
    while True:
        attempt += 1
        trainer, summary, initial_hash = train_attempt(run_dir, snapshot, tokenizer, train, validation,
                                                       raw_validation, config, info, smoke, batch_size, attempt, initial_hash)
        if trainer is not None:
            break
        info["oom_restarts"].append(summary)
        write_json(run_dir / "run.json", info)
        require(batch_size > 1, "OOM at batch=1/accumulation=16; stopping without a completed result.")
        batch_size //= 2
        print(f"OOM: restarting from base with batch={batch_size} / accumulation={16 // batch_size}.", flush=True)
    info["training"] = summary
    epochs = [json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    require([report["epoch"] for report in epochs] == list(range(1, config["epochs"] + 1)), "Missing or duplicate epoch evaluations.")
    require(abs(max(report["model"]["macro_f1"] for report in epochs) - summary["best_validation_macro_f1"]) < 1e-10, "Best checkpoint selection differs from epoch history.")
    output = trainer.predict(validation, metric_key_prefix="best_validation")
    labels = np.asarray(raw_validation["label"])
    require(np.array_equal(labels, output.label_ids), "Final validation order differs.")
    probability = probabilities(output.predictions)
    final = metric_report(metrics(labels, probability), config, smoke, len(labels), "best_checkpoint", metrics(labels, np.ones(len(labels))))
    final["best_checkpoint"] = summary["best_checkpoint"]
    require(abs(final["model"]["macro_f1"] - summary["best_validation_macro_f1"]) < 1e-10, "Reloaded best checkpoint metric differs.")
    write_json(run_dir / "validation_metrics.json", final)
    predicted = (probability >= 0.5).astype(np.int64)
    with (run_dir / "validation_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for identifier, label, p, prediction in zip(raw_validation["id"], labels, probability, predicted):
            stream.write(json.dumps({"id": identifier, "label": int(label), "p_answerable": float(p), "predicted_label": int(prediction)}) + "\n")
    errors = []
    for label in (0, 1):
        for index in np.flatnonzero((labels == label) & (predicted != labels))[:5]:
            row = raw_validation[int(index)]
            errors.append({key: row[key] for key in ("id", "context", "question", "label")} |
                          {"predicted_label": int(predicted[index]), "p_answerable": float(probability[index]),
                           "error_type": "false_answerable" if label == 0 else "false_unanswerable"})
    write_json(run_dir / "validation_errors.json", {"split": "validation", "smoke": smoke,
               "selection": "First up to five errors of each type in saved validation order", "errors": errors})
    info["phase"] = "export"
    write_json(run_dir / "run.json", info)
    before = prediction_probe(trainer.model, tokenizer, raw_validation)
    staging = model_dir.with_name(".incomplete-" + model_dir.name)
    staging.mkdir(parents=True, exist_ok=False)
    trainer.model.save_pretrained(staging, safe_serialization=True)
    tokenizer.save_pretrained(staging)
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    reloaded_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True, use_fast=True)
    reloaded = AutoModelForSequenceClassification.from_pretrained(staging, local_files_only=True)
    after = prediction_probe(reloaded, reloaded_tokenizer, raw_validation)
    require(np.allclose(before, after, atol=1e-5, rtol=1e-4), "Saved/reloaded FP32 logits differ.")
    require(np.array_equal(probabilities(before) >= 0.5, probabilities(after) >= 0.5), "Saved/reloaded classes differ.")
    info["export"] = {"passed": True, "split": "validation", "rows": len(before), "mode": "float32 eval, no autocast",
                      "atol": 1e-5, "rtol": 1e-4, "max_abs_logit_difference": float(np.abs(before - after).max()), "classes_identical": True}
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    require(data_files_sha256() == source_hashes, "Train/validation files changed during the run.")
    require(not model_dir.exists(), "Model destination appeared; nothing overwritten.")
    staging.rename(model_dir)
    info.update(status="completed", phase="completed", completed_at_utc=utc_now(),
                duration_seconds=time.perf_counter() - started, source_files_unchanged=True)
    write_json(run_dir / "run.json", info)
    show_report(final, "BEST CHECKPOINT VALIDATION")
    print(f"\nRUN COMPLETED: {run_dir}\nMODEL AND TOKENIZER: {model_dir}\nEXPORT VERIFIED: {info['export']}", flush=True)


def launch_worker(command, run_dir):
    """Capture native/library stdout and stderr using one worker of this same script."""
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    print(f"Run directory: {run_dir}", flush=True)
    with (run_dir / "stdout.log").open("x", encoding="utf-8") as out, (run_dir / "stderr.log").open("x", encoding="utf-8") as err:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, encoding="utf-8", errors="replace")

        def pump(stream, console, log):
            for line in stream:
                log.write(line)
                log.flush()
                console.write(line)
                console.flush()
            stream.close()

        threads = [threading.Thread(target=pump, args=(child.stdout, sys.stdout, out)),
                   threading.Thread(target=pump, args=(child.stderr, sys.stderr, err))]
        for thread in threads:
            thread.start()
        try:
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            code = child.wait()
        finally:
            for thread in threads:
                thread.join()
    info = read_json(run_dir / "run.json")
    info["process_exit_code"] = code
    if code != 0 or info["status"] != "completed":
        info["status"] = "failed"
        info.setdefault("error", f"Worker exited with code {code}; inspect stderr.log.")
        code = code or 1
    write_json(run_dir / "run.json", info)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "classifier_distilbert.json")
    parser.add_argument("--run-id", help="UTC timestamp followed by model name and seed, e.g. 20261023T120000Z_distilbert_seed42")
    parser.add_argument("--smoke", action="store_true", help="Two epochs on 32 train and 32 validation rows; not a full experiment.")
    parser.add_argument("--local-files-only", action="store_true", help="Require the pinned base model in the project cache; otherwise it may be downloaded on first training.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_dir = None
    try:
        require(0 <= args.seed < 2**32, "Seed must be an unsigned 32-bit integer.")
        run_id = args.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_distilbert_seed{args.seed}" + ("_smoke" if args.smoke else ""))
        require(re.fullmatch(r"\d{8}T\d{6}\d{0,6}Z_[A-Za-z0-9_-]+_seed" + str(args.seed) + r"(?:_smoke)?", run_id), "run_id must contain a UTC timestamp, model name and matching seed.")
        run_dir = OUTPUTS / run_id
        if args.worker:
            info = read_json(run_dir / "run.json")
            require(info["status"] == "starting" and info["config"]["seed"] == args.seed,
                    "Worker requires a newly reserved matching run; existing results are not overwritten.")
            try:
                with training_lock(run_dir):
                    initialize_ml(info["local_files_only"])
                    run_experiment(run_dir, info, args.smoke)
            except BaseException as exc:
                info.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at_utc=utc_now())
                write_json(run_dir / "run.json", info)
                raise
            return 0
        config_path = args.config if args.config.is_absolute() else ROOT / args.config
        config = read_json(config_path)
        config["seed"] = args.seed
        require(config["model_id"] == "distilbert/distilbert-base-uncased" and config["revision"] == "12040accade4e8a0f71eabdb258fecc2e7e948be", "Use the fixed base checkpoint/revision.")
        require(config["epochs"] == 2 and config["max_length"] == 512 and config["dataloader_num_workers"] == 0 and config["threshold"] == 0.5,
                "This protocol requires 2 epochs, max_length=512, workers=0 and threshold=0.5.")
        require(config["train_batch_size"] in (1, 2, 4, 8, 16) and config["train_batch_size"] * config["gradient_accumulation_steps"] == 16,
                "Train batch * accumulation must equal 16.")
        require((PROCESSED / "train").is_dir() and (PROCESSED / "validation").is_dir()
                and (ROOT / "outputs" / "split_summary.json").is_file(),
                "Prepared data are missing. Run download_dataset.py and prepare_splits.py before training.")
        model_dir = ROOT / "models" / "classifier" / run_id
        require(not run_dir.exists() and not model_dir.exists(), "Run/model directory already exists; choose a new timestamped run_id.")
        run_dir.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--seed", str(args.seed), "--run-id", run_id]
        if args.smoke:
            command.append("--smoke")
        if args.local_files_only:
            command.append("--local-files-only")
        info = {"schema": "answerability-classifier-run-v1", "run_id": run_id, "status": "starting",
                "purpose": "smoke" if args.smoke else "full", "config": config, "split_seed": 42,
                "started_at_utc": utc_now(), "model_directory": str(model_dir), "evaluation_label": DEV,
                "test_evaluated": False, "local_files_only": args.local_files_only, "command": command}
        write_json(run_dir / "run.json", info)
        return launch_worker(command, run_dir)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        if args.worker:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
