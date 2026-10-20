"""Prepare reproducible answerability splits using local SQuAD 2.0 only.

Never print test questions, contexts, or answers. No model training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

from datasets import DatasetDict, disable_progress_bars, load_from_disk


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "squad_v2"
DESTINATION = ROOT / "data" / "processed" / "squad_v2_answerability"
OUTPUTS = ROOT / "outputs"
SOURCE_ID = "rajpurkar/squad_v2"
SEED = 42
SPLITS = ("train", "validation", "test")
METADATA_NAME = "preparation_metadata.json"
SETTINGS = {
    "algorithm_version": 1,
    "seed": SEED,
    "label": "1 if len(answers['text']) > 0 else 0",
    "context_group": "SHA-256(UTF-8(' '.join(NFC(context).split())))",
    "unicode_version": unicodedata.unidata_version,
    "overlap_policy": "Remove all raw train rows whose group occurs anywhere in raw validation.",
    "validation_group_fraction": 0.1,
    "group_split": (
        "Sort remaining groups lexicographically; shuffle using random.Random(42). "
        "Take the first max(1, min(n-1, int(n*0.1+0.5))) groups for validation; "
        "all others for train. Preserve original row order within each split."
    ),
    "test_per_class": 500,
    "test_sampling": (
        "Use a separate random.Random(42). Sample 500 indices without replacement "
        "from label=1, then 500 from label=0, in raw validation row order; "
        "shuffle the combined indices using that same test RNG."
    ),
    "manifest_checksum": "SHA-256 of UTF-8 JSON ID array, ensure_ascii=False, separators=(',', ':'); no newline",
    "classifier_input_fields": ["question", "context"],
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ordered_ids_checksum(ids: list[str]) -> str:
    encoded = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_group(context: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", context).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def raw_file_checksums() -> dict[str, str]:
    """Identify actual source bytes, not just cached datasets fingerprints."""
    result = {}
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                result[path.relative_to(RAW).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    require(bool(result), "No local raw dataset files found; this script never downloads data.")
    return result


def enrich(raw: DatasetDict) -> DatasetDict:
    result = DatasetDict()
    for name, split in raw.items():
        required = {"id", "context", "question", "answers"}
        require(required.issubset(split.column_names), f"Raw {name} lacks required fields.")
        require(
            not {"label", "source_split", "context_group"}.intersection(split.column_names),
            f"Raw {name} already contains derived columns; use the original data.",
        )
        labels, groups = [], []
        for row in split:
            labels.append(int(len(row["answers"]["text"]) > 0))
            groups.append(context_group(row["context"]))
        ids = list(split["id"])
        require(len(set(ids)) == len(ids), f"Duplicate IDs in raw {name}.")
        result[name] = (
            split.add_column("label", labels)
            .add_column("source_split", [name] * len(split))
            .add_column("context_group", groups)
        )
    return result


def build_plan(data: DatasetDict) -> tuple[dict[str, list[int]], dict]:
    train_groups = list(data["train"]["context_group"])
    validation_groups = set(data["validation"]["context_group"])
    shared = set(train_groups) & validation_groups
    kept_indices = [index for index, group in enumerate(train_groups) if group not in shared]
    groups = sorted({train_groups[index] for index in kept_indices})
    require(len(groups) >= 2, "At least two remaining context groups are required for train/validation.")
    random.Random(SEED).shuffle(groups)
    validation_count = max(1, min(len(groups) - 1, int(len(groups) * 0.1 + 0.5)))
    selected_validation = set(groups[:validation_count])
    plan = {
        "train": [index for index in kept_indices if train_groups[index] not in selected_validation],
        "validation": [index for index in kept_indices if train_groups[index] in selected_validation],
    }
    labels = list(data["validation"]["label"])
    rng = random.Random(SEED)
    test_indices = []
    for label in (1, 0):
        candidates = [index for index, value in enumerate(labels) if value == label]
        require(len(candidates) >= 500, f"Raw validation has fewer than 500 examples with label={label}.")
        test_indices.extend(rng.sample(candidates, 500))
    rng.shuffle(test_indices)
    plan["test"] = test_indices
    details = {
        "raw_context_group_overlap": len(shared),
        "excluded_train_groups": len(shared),
        "excluded_train_rows": len(train_groups) - len(kept_indices),
        "remaining_raw_train_rows": len(kept_indices),
        "remaining_raw_train_groups": len(groups),
        "unused_raw_validation_rows": len(labels) - len(test_indices),
        "all_remaining_raw_train_rows_assigned_once": (
            sorted(plan["train"] + plan["validation"]) == kept_indices
        ),
        "test_indices_unique": len(set(test_indices)) == len(test_indices),
    }
    require(details["all_remaining_raw_train_rows_assigned_once"], "Train/validation coverage failed.")
    require(details["test_indices_unique"], "Test sampling produced repeated indices.")
    return plan, details


def materialize(data: DatasetDict, plan: dict[str, list[int]]) -> DatasetDict:
    # Keep all transformations in memory: never write datasets cache files into raw/.
    return DatasetDict({
        name: data["validation" if name == "test" else "train"]
        .select(plan[name], keep_in_memory=True)
        .flatten_indices(keep_in_memory=True)
        for name in SPLITS
    })


def audit(data: DatasetDict) -> tuple[dict, dict]:
    require(list(data) == list(SPLITS), "Expected exactly train, validation, test in that order.")
    statistics, id_sets, group_sets = {}, {}, {}
    for name, split in data.items():
        require(len(split) > 0, f"Empty output split: {name}.")
        ids, groups, classes = set(), set(), Counter()
        empty_context = empty_question = label_errors = group_errors = source_errors = 0
        for row in split:
            ids.add(row["id"])
            groups.add(row["context_group"])
            classes[row["label"]] += 1
            label_errors += int(type(row["label"]) is not int or row["label"] != int(bool(row["answers"]["text"])))
            group_errors += int(row["context_group"] != context_group(row["context"]))
            source_errors += int(row["source_split"] != ("validation" if name == "test" else "train"))
            empty_context += int(not row["context"].strip())
            empty_question += int(not row["question"].strip())
        statistics[name] = {
            "rows": len(split),
            "answerable": classes[1],
            "unanswerable": classes[0],
            "unique_context_groups": len(groups),
            "unique_ids": len(ids),
            "duplicate_id_extra_rows": len(split) - len(ids),
            "empty_context": empty_context,
            "empty_question": empty_question,
            "label_mismatches": label_errors,
            "context_group_mismatches": group_errors,
            "source_split_mismatches": source_errors,
            "columns": split.column_names,
            "fingerprint": getattr(split, "_fingerprint", None),
        }
        require(not (len(split) - len(ids)), f"Duplicate IDs in {name}.")
        require(not (empty_context + empty_question), f"Empty context/question found in {name}.")
        require(not (label_errors + group_errors + source_errors), f"Derived column validation failed in {name}.")
        require(classes[0] + classes[1] == len(split), f"Non-binary labels in {name}.")
        id_sets[name], group_sets[name] = ids, groups
    intersections = []
    for left, right in combinations(SPLITS, 2):
        id_overlap = len(id_sets[left] & id_sets[right])
        group_overlap = len(group_sets[left] & group_sets[right])
        intersections.append({"splits": [left, right], "shared_ids": id_overlap, "shared_context_groups": group_overlap})
        require(id_overlap == 0 and group_overlap == 0, f"ID/context group leakage between {left} and {right}.")
    require(statistics["test"]["answerable"] == statistics["test"]["unanswerable"] == 500,
            "Test must contain exactly 500 examples of each class.")
    return statistics, {"passed": True, "intersections": intersections}


def make_manifest(data: DatasetDict) -> dict:
    return {
        "source_id": SOURCE_ID,
        "seed": SEED,
        "checksum_algorithm": SETTINGS["manifest_checksum"],
        "splits": {
            name: {"ids": list(split["id"]), "ids_sha256": ordered_ids_checksum(list(split["id"]))}
            for name, split in data.items()
        },
    }


def verify_equal(actual: DatasetDict, expected: DatasetDict) -> None:
    require(list(actual) == list(expected), "Saved split names/order differ from the deterministic rebuild.")
    for name in SPLITS:
        require(actual[name].features == expected[name].features, f"Saved features differ in {name}.")
        require(list(actual[name]["id"]) == list(expected[name]["id"]), f"Saved IDs/order differ in {name}.")
        require(actual[name].data.table.equals(expected[name].data.table),
                f"Saved values differ from the source-derived data in {name}.")


def prepare(overwrite: bool) -> dict:
    disable_progress_bars()
    require(RAW.is_dir(), "Missing data/raw/squad_v2. Restore the local raw dataset; no download is attempted.")
    print("Loading local raw DatasetDict; test content will not be printed.", flush=True)
    raw_checksums = raw_file_checksums()
    raw = load_from_disk(str(RAW))
    require(isinstance(raw, DatasetDict) and set(raw) == {"train", "validation"},
            "Raw dataset must contain the original train and validation splits only.")
    versions = {"python": platform.python_version(), "datasets": version("datasets"), "pyarrow": version("pyarrow")}
    source = {
        "source_id": SOURCE_ID,
        "raw_file_sha256": raw_checksums,
        "splits": {
            name: {"rows": len(split), "columns": split.column_names, "fingerprint": getattr(split, "_fingerprint", None)}
            for name, split in raw.items()
        },
    }
    identity = {"source": source, "settings": SETTINGS, "versions": versions}
    metadata_path = DESTINATION / METADATA_NAME
    summary_path, manifest_path = OUTPUTS / "split_summary.json", OUTPUTS / "split_manifest.json"
    existing_metadata = None
    if not overwrite:
        if DESTINATION.exists():
            require(metadata_path.is_file(), "Existing processed data lack preparation metadata. Use --overwrite for intentional recreation.")
            existing_metadata = read_json(metadata_path)
            require(existing_metadata["identity"] == identity,
                    "Existing data use different source/settings/versions. Nothing overwritten; use --overwrite intentionally.")
        else:
            require(not summary_path.exists() and not manifest_path.exists(),
                    "Split reports already exist without processed data. Nothing overwritten; use --overwrite intentionally.")

    enriched = enrich(raw)
    plan, allocation = build_plan(enriched)
    second_plan, second_allocation = build_plan(enriched)
    require(plan == second_plan and allocation == second_allocation,
            "Repeated construction with seed=42 produced different indices/order.")
    expected = materialize(enriched, plan)
    expected_stats, _ = audit(expected)
    expected_manifest = make_manifest(expected)
    checksums = {name: item["ids_sha256"] for name, item in expected_manifest["splits"].items()}

    mode = "reuse" if existing_metadata is not None else ("overwrite" if overwrite else "create")
    staging = None
    if mode == "reuse":
        require(existing_metadata["manifest_checksums"] == checksums,
                "Stored metadata checksums differ from rebuilt IDs. Use --overwrite intentionally.")
        if manifest_path.exists():
            require(read_json(manifest_path) == expected_manifest,
                    "Existing split_manifest.json differs from rebuilt IDs/order/checksums. Use --overwrite intentionally.")
        if summary_path.exists():
            previous = read_json(summary_path)
            require(previous["identity"] == identity and previous["manifest_checksums"] == checksums,
                    "Existing split_summary.json differs from the source/settings/manifest. Use --overwrite intentionally.")
            require(previous["allocation"] == allocation, "Existing summary allocation differs. Use --overwrite intentionally.")
        loaded = load_from_disk(str(DESTINATION))
        print("Checking existing processed data against a fresh deterministic rebuild.", flush=True)
    else:
        DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".squad_v2_answerability.incomplete-", dir=DESTINATION.parent))
        print("Saving prepared DatasetDict and reopening it for validation.", flush=True)
        expected.save_to_disk(str(staging))
        loaded = load_from_disk(str(staging))

    verify_equal(loaded, expected)
    statistics, validation = audit(loaded)
    require(make_manifest(loaded) == expected_manifest, "Saved manifest differs from deterministic rebuild.")
    # Stats include library fingerprints; compare all substantive fields to the new build.
    for name in SPLITS:
        for key in expected_stats[name]:
            if key != "fingerprint":
                require(statistics[name][key] == expected_stats[name][key], f"Saved statistics differ in {name}.")
        if mode == "reuse" and summary_path.exists():
            require(read_json(summary_path)["splits"][name] == statistics[name],
                    f"Existing summary statistics differ in {name}. Use --overwrite intentionally.")
    require(raw_file_checksums() == raw_checksums, "Raw files changed during preparation; publication stopped.")
    remaining = allocation["remaining_raw_train_rows"]
    summary = {
        "identity": identity,
        "created_at_utc": existing_metadata["created_at_utc"] if existing_metadata else utc_now(),
        "checked_at_utc": utc_now(),
        "run_mode": mode,
        "python_executable": sys.executable,
        "allocation": allocation,
        "actual_remaining_train_row_fractions": {name: len(loaded[name]) / remaining for name in ("train", "validation")},
        "splits": statistics,
        "manifest_checksums": checksums,
        "checks": {
            **validation,
            "disk_read_successful": True,
            "all_saved_fields_equal_source_derived_fields": True,
            "rebuild_same_seed_same_ids_and_order": True,
            "saved_ids_and_order_match_rebuild": True,
            "raw_files_unchanged": True,
            "test_balanced_500_per_class": True,
        },
    }
    if staging is not None:
        write_json(staging / METADATA_NAME, {
            "identity": identity, "created_at_utc": summary["created_at_utc"], "manifest_checksums": checksums,
        })
        # Close memory maps before moving folders on Windows; retain an old copy on --overwrite.
        del loaded
        backup = None
        if DESTINATION.exists():
            require(overwrite, "Destination appeared during preparation; no files overwritten.")
            backup = DESTINATION.with_name(DESTINATION.name + ".backup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
            DESTINATION.rename(backup)
        try:
            staging.rename(DESTINATION)
        except OSError:
            if backup is not None:
                backup.rename(DESTINATION)
            raise
        # Check the final published path too, independently of the staging path.
        published = load_from_disk(str(DESTINATION))
        verify_equal(published, expected)
        if backup is not None:
            summary["previous_dataset_backup"] = str(backup)
            print(f"Previous processed data retained at {backup}", flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, expected_manifest)
    write_json(summary_path, summary)
    print(f"Complete: mode={mode}; raw files unchanged; all checks passed.", flush=True)
    for name, stats in statistics.items():
        print(f"{name}: rows={stats['rows']}, answerable={stats['answerable']}, "
              f"unanswerable={stats['unanswerable']}, context_groups={stats['unique_context_groups']}", flush=True)
    print(f"Excluded raw train: {allocation['excluded_train_rows']} rows / {allocation['excluded_train_groups']} groups.", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Intentionally recreate processed splits/reports; keep the previous dataset as a backup.")
    args = parser.parse_args()
    try:
        prepare(args.overwrite)
        return 0
    except Exception as exc:
        # Avoid including source rows or third-party exception text that could expose test content.
        detail = str(exc) if isinstance(exc, ValueError) else (
            "Check local dataset integrity, metadata JSON, permissions and free disk space. "
            "Incomplete staging folders are retained for inspection."
        )
        print(f"ERROR: {type(exc).__name__}: {detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
