"""Read saved classifier reports and compare completed runs without loading models or datasets."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = "Development evaluation on validation; test not evaluated"
INITIAL = "pretrained encoder with randomly initialized classification head, before task fine-tuning."
METRIC_NAMES = ("accuracy", "balanced_accuracy", "macro_f1", "roc_auc",
                "precision_0", "recall_0", "f1_0", "precision_1", "recall_1", "f1_1")


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing required report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    if path.exists():
        raise ValueError(f"Comparison output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Paths must remain inside the project.")
    return path


def metric_values(report):
    result = {name: report[name] for name in METRIC_NAMES[:4]}
    for label in ("0", "1"):
        for name, original in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score")):
            result[f"{name}_{label}"] = report["per_class"][label][original]
    return result


def table(rows):
    print(f"{'Stage':<23} " + " ".join(f"{('balanced_acc' if name == 'balanced_accuracy' else name):>12}" for name in METRIC_NAMES))
    for stage, report in rows:
        values = metric_values(report)
        print(f"{stage:<23} " + " ".join(f"{values[name]:12.6f}" for name in METRIC_NAMES))


def confusion(report):
    matrix = report["confusion_matrix"]
    print("Confusion matrix: rows=true, columns=predicted; 0=unanswerable, 1=answerable")
    print("                 predicted 0   predicted 1")
    print(f"true 0          {matrix[0][0]:12d}  {matrix[0][1]:12d}")
    print(f"true 1          {matrix[1][0]:12d}  {matrix[1][1]:12d}")


def show_report(report, heading):
    print(f"\n{heading}")
    print(DEV)
    if report.get("smoke"):
        print("SMOKE ONLY: small technical subset, excluded from experiment statistics.")
    if report.get("description"):
        print(report["description"])
    print(f"Validation rows: {report.get('rows', sum(map(sum, report['model']['confusion_matrix'])))}; threshold: 0.5")
    table([("model", report["model"])] + ([("always-answerable", report["always_answerable"])] if "always_answerable" in report else []))
    confusion(report["model"])


def show_error(directory, index):
    report = read_json(directory / "validation_errors.json")
    if report.get("split") != "validation":
        raise ValueError("Only validation error reports are permitted.")
    errors = report["errors"]
    if not 0 <= index < len(errors):
        raise ValueError(f"Error index {index} unavailable; report contains {len(errors)} examples.")
    row = errors[index]
    print(DEV)
    if report.get("smoke"):
        print("SMOKE ONLY: error from a small technical validation subset.")
    print("REAL VALIDATION ERROR; full text from the saved error report")
    print(f"ID: {row['id']}\nTrue label: {row['label']}\nPredicted label: {row['predicted_label']}\np(answerable): {row['p_answerable']:.8f}")
    print(f"\nQUESTION\n{row['question']}\n\nCONTEXT\n{row['context']}")


def read_epochs(directory):
    path = directory / "epoch_metrics.jsonl"
    if path.is_file():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [read_json(path) for path in sorted(directory.glob("epoch_*_validation_metrics.json"))]


def read_metadata(directory):
    if (directory / "run.json").is_file():
        return read_json(directory / "run.json")
    # Read historical layouts without rewriting them or fabricating missing initial metrics.
    status = read_json(directory / "status.json")
    info = {**status, "legacy": True}
    for key, name in (("config", "requested_config.json"), ("versions", "versions.json"),
                      ("data", "data_signature.json"), ("export", "save_load_verification.json")):
        if (directory / name).is_file():
            info[key] = read_json(directory / name)
    for name in ("training_summary.json", "full/training_summary.json"):
        if (directory / name).is_file():
            info["training"] = read_json(directory / name)
            break
    initial_path = directory / "initial_validation_metrics.json"
    if initial_path.is_file():
        info["initial_weights_check"] = read_json(initial_path).get("initialization_check", {})
    info.setdefault("purpose", info.get("training", {}).get("stage", "unknown"))
    return info


def show_run(directory):
    info = read_metadata(directory)
    print(DEV)
    print(f"Run: {directory.name}; status={info['status']}; purpose={info.get('purpose', 'unknown')}")
    print(f"Model: {info.get('config', {}).get('model_id', 'see historical provenance')}; seed={info.get('config', {}).get('seed', 'unknown')}")
    initial_path = directory / "initial_validation_metrics.json"
    if initial_path.is_file():
        show_report(read_json(initial_path), "INITIAL VALIDATION")
    else:
        print("Initial validation metrics are not present; they are not reconstructed.")
    for report in read_epochs(directory):
        show_report(report, f"{report['stage'].upper()} VALIDATION")
    final_path = directory / "validation_metrics.json"
    if final_path.is_file():
        show_report(read_json(final_path), "BEST CHECKPOINT VALIDATION")
    elif info["status"] == "completed":
        raise FileNotFoundError(f"Completed run lacks {final_path}")
    else:
        print("Final validation metrics are not available: this run has not completed.")
    for key, title in (("training", "TRAINING SUMMARY"), ("export", "EXPORT VERIFICATION")):
        if key in info:
            print(f"\n{title}\n{json.dumps(info[key], indent=2)}")
    history = directory / "history.jsonl"
    if history.is_file():
        lines = history.read_text(encoding="utf-8").splitlines()
        if lines:
            print(f"\nLATEST HISTORY EVENT\n{lines[-1]}")


def completed_run(directory):
    info = read_metadata(directory)
    if info["status"] != "completed" or info.get("purpose") != "full":
        raise ValueError("not a completed full experiment; smoke and interrupted runs are excluded")
    requested_epochs = info["config"]["epochs"]
    if not isinstance(requested_epochs, int) or requested_epochs < 1:
        raise ValueError("run configuration must specify a positive integer epoch count")
    if not info.get("export", {}).get("passed") or info.get("training", {}).get("completed_epochs", 0) < requested_epochs:
        raise ValueError(f"training/export verification is incomplete: expected {requested_epochs} epochs")
    initial = read_json(directory / "initial_validation_metrics.json")
    epochs = read_epochs(directory)
    final = read_json(directory / "validation_metrics.json")
    check = info.get("initial_weights_check", {})
    if not check.get("unchanged") or check.get("optimizer_steps", initial.get("optimizer_steps")) != 0:
        raise ValueError("initial evaluation must demonstrate unchanged weights before the first optimizer step")
    expected_epochs = list(range(1, requested_epochs + 1))
    if [report.get("epoch") for report in epochs] != expected_epochs:
        raise ValueError(f"expected saved evaluation after epochs {expected_epochs}")
    if info["training"]["effective_batch_size"] != 16:
        raise ValueError("actual effective batch differs from 16")
    if abs(max(report["model"]["macro_f1"] for report in epochs) - final["model"]["macro_f1"]) > 1e-10:
        raise ValueError("final metrics do not match the best epoch macro-F1")
    for report in [initial, *epochs, final]:
        if (report.get("evaluation_split") != "validation" or report.get("smoke")
                or report.get("test_evaluated", report.get("test_accessed", False))
                or report["rows"] != info["data"]["rows"]["validation"]
                or report["training_seed"] != info["config"]["seed"] or report["model"]["threshold"] != 0.5):
            raise ValueError("evaluation scope, seed, threshold or row count differs from the protocol")
        if not all(math.isfinite(value) for value in metric_values(report["model"]).values()):
            raise ValueError("nonfinite metric")
    return {"directory": directory, "info": info, "initial": initial, "epochs": epochs, "final": final}


def compare_runs(directories, destination):
    runs = []
    print(DEV)
    for directory in directories:
        try:
            runs.append(completed_run(directory))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"EXCLUDED: {directory}: {exc}")
    if len(runs) != 3 or len({run['info']['config']['seed'] for run in runs}) != 3:
        raise ValueError("Three completed full runs with distinct training seeds are required; no n=3 statistics produced.")
    reference = runs[0]
    for run in runs[1:]:
        for key in ("data", "versions"):
            if run["info"][key] != reference["info"][key]:
                raise ValueError(f"Runs differ in {key}; comparison refused.")
        settings = lambda item: {key: value for key, value in item["info"]["config"].items() if key != "seed"}
        if settings(run) != settings(reference):
            raise ValueError("Requested configurations differ beyond training seed.")
        if run["final"]["always_answerable"] != reference["final"]["always_answerable"]:
            raise ValueError("Baseline differs across runs.")
    rows, deltas = [], {}
    for run in sorted(runs, key=lambda item: item["info"]["config"]["seed"]):
        seed = run["info"]["config"]["seed"]
        actual = run["info"]["training"]
        print(f"Seed {seed}: {run['directory'].name}; batch={actual['train_batch_size']}, accumulation={actual['gradient_accumulation_steps']}, precision={actual['precision']}")
        rows.append((f"seed {seed} / initial", run["initial"]["model"]))
        rows += [(f"seed {seed} / epoch {i}", report["model"]) for i, report in enumerate(run["epochs"], 1)]
        rows.append((f"seed {seed} / best", run["final"]["model"]))
        before, after = metric_values(run["initial"]["model"]), metric_values(run["final"]["model"])
        deltas[str(seed)] = {key: after[key] - before[key] for key in METRIC_NAMES}
    rows.append(("always-answerable", reference["final"]["always_answerable"]))
    table(rows)
    summary = {}
    print("\nBEST CHECKPOINT METRICS: n=3; sample SD uses denominator n-1")
    for key in METRIC_NAMES:
        values = [metric_values(run["final"]["model"])[key] for run in runs]
        changes = [deltas[str(run["info"]["config"]["seed"])][key] for run in runs]
        summary[key] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                        "delta_mean": statistics.mean(changes), "delta_sample_std": statistics.stdev(changes)}
        result = summary[key]
        print(f"{key:20} mean={result['mean']:.6f} sample_sd={result['sample_std']:.6f} "
              f"delta_mean={result['delta_mean']:+.6f} delta_sample_sd={result['delta_sample_std']:.6f}")
    for seed, changes in sorted(deltas.items()):
        print(f"\nSeed {seed}: best minus initial: " + ", ".join(f"{key}={value:+.6f}" for key, value in changes.items()))
    for run in sorted(runs, key=lambda item: item["info"]["config"]["seed"]):
        print(f"\nSeed {run['info']['config']['seed']}: best checkpoint")
        confusion(run["final"]["model"])
    note = "Sample SD describes training-seed variation, not a confidence interval. All runs reuse the same validation examples; confusion matrices are not pooled."
    print(note)
    write_json(destination, {"evaluation_label": DEV, "n": 3, "run_directories": [str(run["directory"]) for run in runs],
                             "training_seeds": [run["info"]["config"]["seed"] for run in runs],
                             "statistics": summary, "changes_from_initial": deltas, "interpretation": note})
    print(f"Comparison saved: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("show", "error"):
        p = commands.add_parser(name)
        p.add_argument("--run", required=True, help="Run directory, relative to project root or absolute.")
        if name == "error":
            p.add_argument("--index", type=int, default=0)
    p = commands.add_parser("compare")
    p.add_argument("--runs", nargs=3, required=True, metavar="RUN")
    p.add_argument("--output", help="New JSON destination; never overwrites an existing comparison.")
    args = parser.parse_args()
    try:
        if args.command == "show":
            show_run(project_path(args.run))
        elif args.command == "error":
            show_error(project_path(args.run), args.index)
        else:
            from datetime import datetime, timezone
            destination = project_path(args.output) if args.output else ROOT / "outputs" / "comparisons" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
            compare_runs([project_path(value) for value in args.runs], destination)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
