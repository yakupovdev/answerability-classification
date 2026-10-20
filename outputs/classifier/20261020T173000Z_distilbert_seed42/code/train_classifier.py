"""Train an answerability filter using only the separate train/validation directories.

Never loads the processed DatasetDict root or the held-out test directory.
"""

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
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, disable_progress_bars, load_from_disk
from huggingface_hub import HfApi, snapshot_download
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
    Trainer, TrainerCallback, TrainingArguments, set_seed,
)

PROCESSED = ROOT / "data" / "processed" / "squad_v2_answerability"
MODEL_DESTINATION = ROOT / "models" / "answerability_distilbert"


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


def trainer_metrics(prediction):
    report = metrics(prediction.label_ids, probabilities(prediction.predictions))
    return {key: report[key] for key in ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc")}


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


class RunLogger(TrainerCallback):
    def __init__(self, directory):
        self.directory = directory

    def on_log(self, args, state, control, logs=None, **kwargs):
        event = {"utc": utc_now(), "step": state.global_step, "epoch": state.epoch, **(logs or {})}
        with (self.directory / "history.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        write_json(self.directory / "progress.json", event)


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


def run_training(stage, train, validation, tokenizer, snapshot, config, precision, run_dir, gpu):
    directory = run_dir / stage
    directory.mkdir(exist_ok=False)
    if stage == "smoke":
        rng = random.Random(config["seed"])
        train = train.select(rng.sample(range(len(train)), min(config["smoke_train_rows"], len(train))), keep_in_memory=True)
        validation = validation.select(rng.sample(range(len(validation)), min(config["smoke_validation_rows"], len(validation))), keep_in_memory=True)
    batch_size = config["train_batch_size"]
    started = time.perf_counter()
    failures = []
    attempt = 0
    while True:
        attempt += 1
        attempt_dir = directory / f"attempt_{attempt}"
        attempt_dir.mkdir()
        accumulation = 16 // batch_size
        model = trainer = None
        try:
            model = fresh_model(snapshot, config["seed"])
            arguments = TrainingArguments(
                output_dir=str(attempt_dir / "checkpoints"),
                num_train_epochs=config["epochs"], learning_rate=config["learning_rate"],
                # transformers 5.20 represents a warmup ratio as fractional warmup_steps.
                weight_decay=config["weight_decay"], warmup_steps=config["warmup_ratio"],
                per_device_train_batch_size=batch_size, gradient_accumulation_steps=accumulation,
                per_device_eval_batch_size=config["eval_batch_size"],
                dataloader_num_workers=config["dataloader_num_workers"],
                seed=config["seed"], data_seed=config["seed"],
                eval_strategy="steps" if stage == "smoke" else "epoch",
                save_strategy="steps" if stage == "smoke" else "epoch",
                eval_steps=config["smoke_steps"] if stage == "smoke" else None,
                save_steps=config["smoke_steps"] if stage == "smoke" else 500,
                max_steps=config["smoke_steps"] if stage == "smoke" else -1,
                logging_steps=1 if stage == "smoke" else config["logging_steps"],
                load_best_model_at_end=True, metric_for_best_model="macro_f1", greater_is_better=True,
                save_total_limit=2, report_to=[], disable_tqdm=True,
                bf16=precision == "bf16", fp16=precision == "fp16", tf32=False,
                optim="adamw_torch", lr_scheduler_type="linear", max_grad_norm=1.0,
                label_smoothing_factor=0.0,
                dataloader_pin_memory=True,
            )
            write_json(attempt_dir / "training_arguments.json", arguments.to_dict())
            trainer = Trainer(
                model=model, args=arguments, train_dataset=train, eval_dataset=validation,
                processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                compute_metrics=trainer_metrics, callbacks=[RunLogger(attempt_dir)],
            )
            require(trainer.model_accepts_loss_kwargs is False, "Trainer would incorrectly normalize DistilBERT accumulation loss.")
            print(f"Starting {stage}: batch={batch_size}, accumulation={accumulation}, precision={precision}", flush=True)
            torch.cuda.reset_peak_memory_stats()
            trained = trainer.train()
            require(trainer.state.global_step > 0, "No optimization steps were performed.")
            if stage == "full":
                require(trainer.state.epoch >= config["epochs"] - 1e-6, "Full training did not complete all requested epochs.")
            require(all(math.isfinite(float(entry["loss"])) for entry in trainer.state.log_history if "loss" in entry), "Nonfinite logged training loss.")
            require(all(math.isfinite(float(entry["grad_norm"])) for entry in trainer.state.log_history if "grad_norm" in entry), "Nonfinite logged gradient norm.")
            trainer.save_state()
            write_json(attempt_dir / "trainer_history.json", {"history": trainer.state.log_history})
            info = {
                "status": "completed", "stage": stage, "smoke_is_not_final_metrics": stage == "smoke",
                "duration_seconds": time.perf_counter() - started,
                "completed_epochs": trainer.state.epoch, "global_steps": trainer.state.global_step,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_validation_macro_f1": trainer.state.best_metric,
                "train_batch_size": batch_size, "gradient_accumulation_steps": accumulation,
                "effective_batch_size": batch_size * accumulation, "precision": precision,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "oom_restarts": failures, "training_metrics": trained.metrics,
                "initialization": "Fresh base checkpoint and seed; never smoke weights or previous attempt weights.",
            }
            write_json(directory / "training_summary.json", info)
            return trainer, info
        except torch.cuda.OutOfMemoryError as exc:
            failures.append({"attempt": attempt, "batch_size": batch_size, "accumulation": accumulation, "error": str(exc)})
            write_json(directory / "oom_restarts.json", {"attempts": failures})
            if batch_size <= 1:
                raise RuntimeError("CUDA out of memory even at batch=1/accumulation=16; training stopped.") from exc
            del trainer, model
            gc.collect()
            torch.cuda.empty_cache()
            batch_size //= 2
            print(f"OOM: restarting {stage} from base with batch={batch_size}; effective batch remains 16.", flush=True)


def prediction_probe(model, tokenizer, raw_validation):
    inputs = tokenizer(raw_validation[:8]["question"], raw_validation[:8]["context"],
                       max_length=512, truncation="only_second", padding=True,
                       return_token_type_ids=False, return_tensors="pt")
    model.eval().to("cuda")
    with torch.inference_mode():
        return model(**{key: value.to("cuda") for key, value in inputs.items()}).logits.float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "classifier_distilbert.json")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_distilbert_seed42")
    parser.add_argument("--stage", choices=("smoke", "full", "all"), default="all")
    parser.add_argument("--local-files-only", action="store_true", help="Use an already cached, commit-pinned base checkpoint without any Hugging Face requests.")
    for name, kind in (("seed", int), ("epochs", float), ("learning-rate", float), ("weight-decay", float),
                       ("warmup-ratio", float), ("train-batch-size", int), ("gradient-accumulation-steps", int),
                       ("eval-batch-size", int), ("max-length", int), ("smoke-steps", int)):
        parser.add_argument("--" + name, type=kind)
    args = parser.parse_args()
    require(args.run_id and all(character.isalnum() or character in "_-" for character in args.run_id), "Unsafe run ID.")
    config = read_json(args.config)
    for name in config:
        override = getattr(args, name, None)
        if override is not None:
            config[name] = override
    require(config["train_batch_size"] in (1, 2, 4, 8, 16) and config["train_batch_size"] * config["gradient_accumulation_steps"] == 16,
            "Train batch * accumulation must equal 16.")
    require(config["max_length"] == 512, "The first experiment requires max_length=512.")
    require(config["dataloader_num_workers"] == 0, "This Windows experiment uses dataloader workers=0.")
    if args.stage in ("all", "full"):
        require(not MODEL_DESTINATION.exists(), "Model directory already exists; choose/archive it explicitly before another full run. Nothing overwritten.")
    run_dir = ROOT / "outputs" / "classifier" / args.run_id
    if args.stage != "full":
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "requested_config.json", config)
    else:
        require(run_dir.is_dir() and read_json(run_dir / "requested_config.json") == config, "Full stage requires the matching previous smoke run ID/config.")
        require(not (run_dir / "full").exists(), "Full results already exist; use a new run ID, not silent overwrite.")
    started = time.perf_counter()
    try:
        disable_progress_bars()
        torch.set_num_threads(8)
        gpu = gpu_probe()
        write_json(run_dir / "gpu_check.json", gpu)
        print(f"CUDA backward passed; GPU={gpu['name']}; precision={gpu['precision']}", flush=True)
        versions = {name: version(name) for name in ("torch", "transformers", "accelerate", "scikit-learn", "datasets", "numpy", "tokenizers", "huggingface-hub")}
        versions["python"] = platform.python_version()
        write_json(run_dir / "versions.json", versions)
        write_json(run_dir / "installation_commands.json", {"commands": [
            "& '.\\.venv\\Scripts\\python.exe' -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128",
            "& '.\\.venv\\Scripts\\python.exe' -m pip install transformers accelerate scikit-learn",
            "& '.\\.venv\\Scripts\\python.exe' -m pip check",
        ], "system_cuda_toolkit_installed": False, "torchvision_torchaudio_installed": False})
        snapshot_info_path = run_dir / "checkpoint_source.json"
        if args.stage == "full":
            checkpoint = read_json(snapshot_info_path)
            snapshot = checkpoint["snapshot_path"]
        else:
            pinned = re.fullmatch(r"[0-9a-f]{40}", config["revision"]) is not None
            require(not args.local_files_only or pinned, "Offline initialization requires a pinned 40-character commit SHA.")
            revision = config["revision"] if pinned else HfApi().model_info(config["model_id"], revision=config["revision"]).sha
            snapshot = snapshot_download(
                config["model_id"], revision=revision, cache_dir=str(ROOT / ".cache" / "huggingface" / "hub"),
                allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"],
                local_files_only=args.local_files_only,
            )
            checkpoint = {"model_id": config["model_id"], "model_revision": revision, "tokenizer_revision": revision, "snapshot_path": snapshot}
            write_json(snapshot_info_path, checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
        require(tokenizer.is_fast, "Fast tokenizer is required for offset diagnostics.")
        source_checksums = data_files_sha256()
        # Load the permitted directories individually. Do not load the DatasetDict root.
        raw_train = load_from_disk(str(PROCESSED / "train"))
        raw_validation = load_from_disk(str(PROCESSED / "validation"))
        for name, source in (("train", raw_train), ("validation", raw_validation)):
            require(all(int(bool(answer["text"])) == label for answer, label in zip(source["answers"], source["label"])), f"Invalid labels in {name}.")
        signature = {"files_sha256": source_checksums, "rows": {"train": len(raw_train), "validation": len(raw_validation)},
                     "model_revision": checkpoint["model_revision"], "max_length": config["max_length"],
                     "input_fields": ["question", "context"], "label_only_for_supervision": True, "test_accessed": False}
        if args.stage == "full":
            require(read_json(run_dir / "data_signature.json") == signature, "Train/validation or tokenizer identity changed after smoke run.")
            train = load_from_disk(str(run_dir / "tokenized" / "train"))
            validation = load_from_disk(str(run_dir / "tokenized" / "validation"))
        else:
            write_json(run_dir / "data_signature.json", signature)
            diagnostics = {}
            train, diagnostics["train"] = tokenize_and_diagnose(raw_train, tokenizer, config, "train")
            validation, diagnostics["validation"] = tokenize_and_diagnose(raw_validation, tokenizer, config, "validation")
            write_json(run_dir / "truncation_summary.json", diagnostics)
            train.save_to_disk(str(run_dir / "tokenized" / "train"))
            validation.save_to_disk(str(run_dir / "tokenized" / "validation"))
        if args.stage in ("smoke", "all"):
            write_json(run_dir / "accumulation_verification.json", verify_accumulation(snapshot, tokenizer, train, config["seed"]))
            trainer, smoke = run_training("smoke", train, validation, tokenizer, snapshot, config, gpu["precision"], run_dir, gpu)
            require(math.isfinite(smoke["best_validation_macro_f1"]), "Smoke evaluation failed.")
            write_json(run_dir / "smoke" / "status.json", {"status": "completed", "not_final_metrics": True, "completed_at_utc": utc_now()})
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            if args.stage == "smoke":
                write_json(run_dir / "status.json", {"status": "smoke_completed", "test_accessed": False})
                print(f"Smoke completed separately; full training not started. Run ID: {args.run_id}", flush=True)
                return
        require(read_json(run_dir / "smoke" / "status.json")["status"] == "completed", "A successful separate smoke run is required.")
        require(read_json(run_dir / "accumulation_verification.json")["passed"], "Gradient accumulation verification is required.")
        write_json(run_dir / "status.json", {"status": "full_running", "started_at_utc": utc_now(), "test_accessed": False})
        trainer, training = run_training("full", train, validation, tokenizer, snapshot, config, gpu["precision"], run_dir, gpu)
        output = trainer.predict(validation, metric_key_prefix="final_validation")
        probability = probabilities(output.predictions)
        labels = np.asarray(raw_validation["label"], dtype=np.int64)
        report = {"model": metrics(labels, probability), "always_answerable": metrics(labels, np.ones(len(labels))),
                  "evaluation_split": "validation", "checkpoint_selection_split": "validation", "test_accessed": False,
                  "interpretation": "Development metrics of the classifier only, not independent final evaluation or a test of H1/H2."}
        write_json(run_dir / "validation_metrics.json", report)
        predicted = (probability >= 0.5).astype(np.int64)
        with (run_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for identifier, label, prob, prediction in zip(raw_validation["id"], labels, probability, predicted):
                stream.write(json.dumps({"id": identifier, "label": int(label), "p_answerable": float(prob), "predicted_label": int(prediction)}) + "\n")
        errors = []
        for true_class in (0, 1):
            indices = np.flatnonzero((labels == true_class) & (predicted != labels))[:5]
            for index in indices:
                row = raw_validation[int(index)]
                errors.append({key: row[key] for key in ("id", "context", "question", "label")} | {
                    "p_answerable": float(probability[index]), "predicted_label": int(predicted[index]),
                    "error_type": "false_answerable" if true_class == 0 else "false_unanswerable",
                })
        write_json(run_dir / "validation_errors.json", {"split": "validation", "selection": "First up to five errors of each type in saved validation order", "errors": errors})
        before = prediction_probe(trainer.model, tokenizer, raw_validation)
        staging = MODEL_DESTINATION.with_name(".answerability_distilbert.incomplete-" + args.run_id)
        staging.mkdir(parents=True, exist_ok=False)
        trainer.model.save_pretrained(staging, safe_serialization=True)
        tokenizer.save_pretrained(staging)
        provenance = {"run_id": args.run_id, "run_directory": str(run_dir), "checkpoint": checkpoint, "config": config,
                      "actual_training": training, "versions": versions, "gpu": gpu, "test_accessed": False}
        write_json(staging / "training_provenance.json", provenance)
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        reloaded_tokenizer = AutoTokenizer.from_pretrained(staging, local_files_only=True, use_fast=True)
        reloaded = AutoModelForSequenceClassification.from_pretrained(staging, local_files_only=True)
        after = prediction_probe(reloaded, reloaded_tokenizer, raw_validation)
        require(np.allclose(before, after, atol=1e-5, rtol=1e-4), "Saved/reloaded model logits failed numerical equivalence.")
        require(np.array_equal(probabilities(before) >= 0.5, probabilities(after) >= 0.5), "Saved/reloaded classes differ.")
        verification = {"passed": True, "split": "validation", "rows": len(before), "mode": "float32 eval, no autocast",
                        "atol": 1e-5, "rtol": 1e-4, "max_abs_logit_difference": float(np.abs(before - after).max()), "classes_identical": True}
        write_json(run_dir / "save_load_verification.json", verification)
        del reloaded
        gc.collect()
        require(not MODEL_DESTINATION.exists(), "Model destination appeared during training; nothing overwritten.")
        staging.rename(MODEL_DESTINATION)
        require(data_files_sha256() == source_checksums, "Train/validation source files changed during training.")
        write_json(run_dir / "status.json", {"status": "completed", "completed_at_utc": utc_now(), "total_duration_seconds": time.perf_counter() - started,
                   "training": training, "model_directory": str(MODEL_DESTINATION), "source_files_unchanged": True, "test_accessed": False})
        print(json.dumps({"run_id": args.run_id, "training": training, "validation_metrics": report, "model_directory": str(MODEL_DESTINATION)}, indent=2), flush=True)
    except BaseException as exc:
        write_json(run_dir / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                   "duration_seconds": time.perf_counter() - started, "test_accessed": False})
        raise


if __name__ == "__main__":
    main()
