# Answerability classification

A binary classifier takes **question + context** and predicts whether the
provided text contains enough information to answer: `1` means answerable,
`0` means unanswerable. DistilBERT is fine-tuned for this task; the current
pipeline does not use a generative LLM.

## Project structure

```text
configs/
  classifier_distilbert.json  # model and training hyperparameters
  data_split.json             # split algorithm, reference sizes and ID SHA-256 hashes
scripts/
  check_environment.py        # Windows hardware, library and CUDA diagnostics
  download_dataset.py         # download/reuse and audit the source dataset
  prepare_splits.py           # create or verify the fixed experimental splits
  train_classifier.py         # train, evaluate, export and verify a classifier
  report_results.py           # inspect reports/errors and compare training seeds
data/raw/                     # original SQuAD 2.0
data/processed/               # experimental train/validation/test
models/                       # saved models
outputs/                      # reports, predictions and logs
requirements.txt
```

Working directories are created automatically. Data, models, outputs, the
`.cache/` cache, the `.venv/` environment and secrets are excluded from Git.
The repository tracks source code, configurations, dependencies and documentation.

## Installation

Use Python 3.12 and a compatible NVIDIA GPU for training. Run commands from the
repository root. Virtual environment activation and PowerShell Execution Policy
changes are unnecessary. Skip environment creation if `.venv` already exists.
Create a separate environment on each OS; a Windows environment cannot be
reused on Linux.

### Windows / PowerShell

```powershell
# New environment only; requires the Python Launcher:
py -3.12 -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

### Linux / Bash

Python 3.12 must include `venv` support.

```bash
# New environment only:
python3.12 -m venv .venv
./.venv/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
./.venv/bin/python -m pip install -r requirements.txt
```

Dependency versions remain pinned in `requirements.txt`. The Linux commands
have not been executed on Linux as part of this documentation update.

## Environment check

### Windows / PowerShell

```powershell
& '.\.venv\Scripts\python.exe' scripts/check_environment.py
```

This writes `environment_check.json` to a new directory under `outputs/audits/`.

### Linux / Bash

`check_environment.py` currently uses Windows registry and memory APIs and
cannot run on Linux. Use these terminal checks instead; they do not create
the Windows script's JSON report:

```bash
lscpu
free -h
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo 'nvidia-smi is not on PATH.'
fi
./.venv/bin/python - <<'PY'
import platform
from importlib.metadata import PackageNotFoundError, version

print('Python:', platform.python_version())
for name in ('torch', 'transformers', 'scikit-learn', 'accelerate'):
    try:
        print(name + ':', version(name))
    except PackageNotFoundError:
        print(name + ': not installed')

try:
    import torch
except ImportError as exc:
    print('PyTorch import failed:', exc)
else:
    print('PyTorch CUDA build:', torch.version.cuda)
    print('CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        x = torch.ones((64, 64), device='cuda')
        result = x @ x
        torch.cuda.synchronize()
        assert bool((result == 64).all().item()), 'GPU tensor check failed'
        print('GPU tensor check: passed')
PY
```

The CUDA version shown by `nvidia-smi` describes driver compatibility. It does
not prove that CUDA Toolkit is installed or that PyTorch can access the GPU.

## Data preparation

### Windows / PowerShell

```powershell
& '.\.venv\Scripts\python.exe' -u scripts/download_dataset.py --summary-only
& '.\.venv\Scripts\python.exe' -u scripts/prepare_splits.py
```

### Linux / Bash

```bash
./.venv/bin/python -u scripts/download_dataset.py --summary-only
./.venv/bin/python -u scripts/prepare_splits.py
```

The download script fetches [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2)
only if `data/raw/squad_v2/` is missing, then reuses the local copy. The audit
reports split sizes, class counts, empty inputs and duplicate IDs.

The dataset authors' original train/validation splits differ from our
experimental splits. Our train and validation are derived from the original
train by context groups; test contains 500 answerable and 500 unanswerable
examples sampled from the original validation.
`context_group = SHA-256(UTF-8(' '.join(NFC(context).split())))`.
Groups are sorted and shuffled with **split seed=42**; approximately 10% of
groups are assigned to validation. Groups overlapping the original validation
are excluded from the original train. Row order within our train/validation
is preserved. The complete test sampling algorithm is recorded in
`configs/data_split.json`.

That configuration contains reference sizes and SHA-256 hashes of **ordered**
ID lists. During preparation and verification, actual results are checked
against these references before saving the prepared dataset. A mismatch
reports expected/actual values and possible causes, stops with an error and
does not update the reference automatically. Matching results are saved as
`outputs/split_summary.json`, `outputs/split_manifest.json` and
`data/processed/squad_v2_answerability/`.

To verify existing splits without modifying the data or manifest:

```powershell
& '.\.venv\Scripts\python.exe' -u scripts/prepare_splits.py --check-only
```

```bash
./.venv/bin/python -u scripts/prepare_splits.py --check-only
```

## Training

The planned training seeds are **42, 43 and 44**; the split seed always remains
**42**. All three runs use `epochs=3` from the shared configuration. Run them
sequentially. The commands assign run IDs for later report viewing and
comparison; keep the variables in the same shell session, or replace them
later with the actual saved directory names.

### Windows / PowerShell

```powershell
$run42 = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_distilbert_seed42'
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --seed 42 --run-id $run42

$run43 = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_distilbert_seed43'
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --seed 43 --run-id $run43

$run44 = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_distilbert_seed44'
& '.\.venv\Scripts\python.exe' -u scripts/train_classifier.py --seed 44 --run-id $run44
```

### Linux / Bash

```bash
run42="$(date -u +%Y%m%dT%H%M%SZ)_distilbert_seed42"
./.venv/bin/python -u scripts/train_classifier.py --seed 42 --run-id "$run42"

run43="$(date -u +%Y%m%dT%H%M%SZ)_distilbert_seed43"
./.venv/bin/python -u scripts/train_classifier.py --seed 43 --run-id "$run43"

run44="$(date -u +%Y%m%dT%H%M%SZ)_distilbert_seed44"
./.venv/bin/python -u scripts/train_classifier.py --seed 44 --run-id "$run44"
```

The first run downloads the pinned base checkpoint
[distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be),
revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
The cache is local to the project. Add `--local-files-only` to use an already
populated cache offline. Existing run/model directories are never overwritten.

Protocol: 3 epochs, learning rate=2e-5, weight decay=0.01, warmup=10%,
train batch=8, gradient accumulation=2, eval batch=8, max_length=512,
workers=0, unweighted cross-entropy, threshold=0.5. BF16 is used when the GPU
probe succeeds; actual settings are recorded in `run.json`. On OOM, the batch
size is reduced while keeping effective batch=16, and the attempt restarts
from a fresh seeded initialization. An interrupted attempt is not a result.

The entire validation split is evaluated before the first optimizer step:

> pretrained encoder with randomly initialized classification head, before task fine-tuning.

The pipeline checks unchanged weights and a zero global step. Before training
the same model, RNG state is restored and the seed is set. Validation is
evaluated after every epoch; the best checkpoint is selected by macro-F1.
The model and tokenizer are saved under `models/classifier/<run_id>/` and
verified after reloading in FP32 on validation. Only question/context enter
the model; answers support label derivation and diagnostics, and labels
are used to compute the loss.

## Results and seed comparison

### Windows / PowerShell

```powershell
& '.\.venv\Scripts\python.exe' scripts/report_results.py show --run "outputs/classifier/$run42"
& '.\.venv\Scripts\python.exe' scripts/report_results.py error --run "outputs/classifier/$run42" --index 0
& '.\.venv\Scripts\python.exe' scripts/report_results.py compare --runs "outputs/classifier/$run42" "outputs/classifier/$run43" "outputs/classifier/$run44"
```

### Linux / Bash

```bash
./.venv/bin/python scripts/report_results.py show --run "outputs/classifier/$run42"
./.venv/bin/python scripts/report_results.py error --run "outputs/classifier/$run42" --index 0
./.venv/bin/python scripts/report_results.py compare --runs "outputs/classifier/$run42" "outputs/classifier/$run43" "outputs/classifier/$run44"
```

Each `outputs/classifier/<run_id>/` contains:

- `run.json`: configuration, versions, revision, data identity, checks and status.
- `initial_validation_metrics.json`: evaluation before task fine-tuning.
- `epoch_metrics.jsonl`: validation metrics after each epoch.
- `history.jsonl`: loss, steps and learning rate.
- `validation_metrics.json`: final metrics, confusion matrix and an
  always-answerable baseline.
- `validation_predictions.jsonl` and `validation_errors.json`: predictions
  and actual misclassified validation examples.
- `stdout.log` and `stderr.log`: captured process output.

Report viewing does not start training. Comparison requires three completed,
compatible full runs and shows initial / epoch 1 / epoch 2 / epoch 3 / best
for each seed, the mean and **sample standard deviation, n=3**, and changes
from initial evaluation. Confusion matrices are not pooled: all runs share
the same validation examples. Variation across seeds is not a confidence
interval. Comparison reports are saved under `outputs/comparisons/`.

## Limitations

- Test is reserved for final evaluation after fixing the protocol. Training,
  tuning and error inspection use train/validation only.
- Grouping identical normalized contexts does not guarantee separation
  by article or topic.
- Context is truncated to the pair's 512-token limit, which can hide a gold
  answer. The original label is retained; diagnostics are saved in `run.json`.
- Validation is used for checkpoint selection, so its metrics describe
  development evaluation rather than an independent final evaluation.
  Probability calibration has not been assessed.
- Pinned versions and seeds do not guarantee bitwise-identical computation
  on different hardware. Split membership is verified separately.
