# Answerability classification

A binary classifier takes **question + context** and predicts whether the
provided text contains enough information to answer: `1` means answerable,
`0` means unanswerable. DistilBERT is fine-tuned for this task. A separate
comparison pipeline evaluates a frozen local generator with and without
the classifier gate; the generator is never trained.

## Project structure

```text
configs/
  classifier_distilbert.json  # model and training hyperparameters
  data_split.json             # split algorithm, reference sizes and ID SHA-256 hashes
  llm_comparison.json         # generator, prompts, token budgets and comparison settings
scripts/
  check_environment.py        # Windows hardware, library and CUDA diagnostics
  download_dataset.py         # download/reuse and audit the source dataset
  prepare_splits.py           # create or verify the fixed experimental splits
  train_classifier.py         # train, evaluate, export and verify a classifier
  report_results.py           # inspect reports/errors and compare training seeds
  compare_llm.py              # prepare, validate, freeze and run the paired comparison
  comparison_utils.py         # scoring, identity checks, replay, review and bootstrap
tests/
  test_llm_comparison.py       # synthetic methodological tests; no models or datasets
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

## Paired local LLM comparison

The classifier training and reporting stage is retained. The new comparison
code is prepared for manual execution; preparing code and passing synthetic
tests do not demonstrate generator quality, GPU memory fit or H1/H2.
Existing full classifier reports can be listed without loading model weights.
No classifier retraining or split recreation is required for this stage.

The two modes use exactly the same ordered examples and frozen generator input:

- **Baseline:** question + context -> Qwen -> short answer or `NO_ANSWER`.
- **Filtered:** classifier probability >= threshold -> the same baseline answer;
  otherwise return abstention with reason `classifier_gate`.

This implementation uses **paired replay**: greedy baseline generation is
performed once, and passed answers are reused byte-for-byte. Replay is not
a measurement of gated inference latency or speedup. The classifier runs in
one process, saves its scores and exits before a second process loads Qwen.

The candidate is [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct),
loaded through `AutoModelForCausalLM` and its official tokenizer. The preparation
command resolves the requested revision to a commit SHA, downloads safetensors
and tokenizer files into the project cache, hashes them and saves a lock.
All subsequent inference commands require `--local-files-only` and work offline.
There is no automatic precision change, CPU offload, quantization or smaller
model fallback. Start with batch=1, BF16, 2048 total tokens and 64 reserved
answer tokens. Memory fit on an 8 GB GPU must be established by the manual smoke.
If BF16 is unsupported, the command stops rather than selecting another dtype.

Compatibility was checked against locally installed Python 3.12.2,
PyTorch 2.11.0+cu128, Transformers 5.20.2, datasets 5.1.0,
huggingface_hub 2.2.0, accelerate 1.15.0 and tokenizers 0.23.3.
Importing PyTorch before Transformers succeeded and exposed `Qwen2ForCausalLM`,
`AutoTokenizer` and `GenerationConfig`; no new package installation was needed.
A Transformers-first import hit a Windows scikit-learn DLL error, so the new
model-loading processes import PyTorch first. API import success is not a GPU
inference check. The implementation follows the official
[model loading API](https://huggingface.co/docs/transformers/v5.20.2/en/main_classes/model),
[generation API](https://huggingface.co/docs/transformers/v5.20.2/en/main_classes/text_generation),
[chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
and [Hub download/revision handling](https://huggingface.co/docs/huggingface_hub/guides/download).

### Manual PowerShell sequence

Run from the repository root with the existing environment. These commands
download and run models only when you execute them. Stop if any command fails;
do not proceed to protocol freeze or test after an unsuccessful validation run.

1. List eligible classifiers. Selection is the highest validation macro-F1
   among completed **full, 3-epoch** runs with seeds 42/43/44 and successful
   export verification. Ties use the smaller seed, then run_id for the same seed.
   Smoke runs are excluded. The explicitly supplied model/run paths must match
   this rule. The existing reports currently select seed 43
   (`20261022T205534158Z_distilbert_seed43`, macro-F1=0.8280648861032631).

```powershell
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py classifiers
& '.\.venv\Scripts\python.exe' scripts/check_environment.py

$classifierRun = 'outputs/classifier/20261022T205534158Z_distilbert_seed43'
$classifierModel = 'models/classifier/20261022T205534158Z_distilbert_seed43'
$generatorLock = 'outputs/llm_comparison/qwen2_5_3b_lock.json'
```

2. Prepare the generator once. This is the only new command that accesses
   the Hub. It creates the lock but does not load generator weights onto GPU
   or run inference. A pre-existing lock is never overwritten; choose a new
   filename for an intentional new preparation.

```powershell
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py prepare-generator --revision main --output $generatorLock

$comparisonArgs = @('--classifier-run', $classifierRun, '--classifier-model', $classifierModel, '--generator-lock', $generatorLock)
```

3. Run a technical smoke on **8 reproducibly sampled validation examples**.
   The subset is chosen without looking at labels, answers, difficulty or
   model performance. It is excluded from final hypothesis statistics.

```powershell
$smoke = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_qwen_validation_smoke'
& '.\.venv\Scripts\python.exe' -u scripts/compare_llm.py smoke @comparisonArgs --local-files-only --run-id $smoke
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py show --run "outputs/llm_comparison/$smoke"
Get-ChildItem "outputs/llm_comparison/$smoke/*_phase_*.json" | ForEach-Object { Get-Content -LiteralPath $_.FullName }
```

Inspect the phase reports for actual peak allocated/reserved CUDA memory,
precision, effective generation settings and tokenizer metadata. Inspect
`stdout.log`/`stderr.log` if it fails. OOM stops the run. Changing dtype, token
budgets or any other settings requires a **new smoke and development run**
with the revised config before freezing a protocol; never silently resume
an old run with changed settings.

4. Run development evaluation on a **fixed 200-example validation subset**.
   Sampling uses `random.Random(42)`, sorted sampled indices and saved IDs;
   classifier scores and both predictions are persisted in input order.

```powershell
$development = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_qwen_validation_development'
& '.\.venv\Scripts\python.exe' -u scripts/compare_llm.py development @comparisonArgs --local-files-only --run-id $development
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py show --run "outputs/llm_comparison/$development"
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py example --run "outputs/llm_comparison/$development" --index 0
```

The main threshold remains **0.5**. Optional sensitivity analysis is explicit,
uses this completed development run only and does not change the default:

```powershell
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py thresholds --run "outputs/llm_comparison/$development" --thresholds 0.3 0.4 0.5 0.6 0.7 --output "outputs/llm_comparison/$development/threshold_sensitivity.json"
```

5. Freeze the protocol after inspecting validation results and accepting all
   settings. This requires completed matching smoke and development runs,
   records the selection rule, hashes, versions, validation subset, prompts,
   tokenizer/template, actual validated hardware/precision/generation settings,
   threshold, scoring/review rubric and hypotheses, and
   **does not open test**.

Manual review is optional and is not a prerequisite for freeze or test. The new
frozen protocol explicitly records normalized EM as primary scoring and binds
the primary H1/H2 interpretation to automatic paired test outcomes.

```powershell
$protocol = "outputs/llm_comparison/$development/frozen/protocol.json"
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py freeze @comparisonArgs --smoke-run "outputs/llm_comparison/$smoke" --development-run "outputs/llm_comparison/$development" --output $protocol
```

Compatibility includes the complete hashes of comparison source files. If code
has changed since smoke/development (including reporting or review UI code),
strict freeze refuses those old runs. Repeat **both smoke and development**
with the current code and new run IDs; historical runs remain readable. Do not
edit their protocols/hashes or bypass the compatibility check. No manual review
is needed for these repeated validation steps.

For a deliberate non-default threshold, additionally pass `--threshold VALUE`
and `--threshold-report` pointing to the matching validation sensitivity JSON.
The protocol records that explicit validation-based choice. Never select or
revise a threshold using test results.

6. **Final test only, after freeze.** This separate command opens the complete
   fixed test split and requires the protocol and matching models/settings.
   There is no test subset, limit or debug option. Completion requires exactly
   1000 unique IDs in manifest order, 500 examples per class and two valid
   outcomes for every ID. Do not use its results to revise the frozen protocol.

```powershell
$final = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssfffZ') + '_qwen_paired_test'
& '.\.venv\Scripts\python.exe' -u scripts/compare_llm.py test @comparisonArgs --local-files-only --protocol $protocol --run-id $final
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py show --run "outputs/llm_comparison/$final"
```

To resume an interrupted run, repeat its **identical** smoke/development/test
command, keeping the same run_id and adding `--resume`. Existing outputs are
otherwise rejected. Resume checks settings, model hashes, code, versions,
data and ordered IDs; it skips completed IDs and repairs a missing filtered
pair from its saved baseline. A torn final JSONL write can be recovered only
with explicit resume; discarded tail bytes are recorded in `recovery.jsonl`.
Other corruption or duplicate IDs stop execution. A `.run.lock` prevents two
processes writing the same run; inspect its PID before removing a stale lock.

### Inputs, scoring and inference

The generator uses the official `apply_chat_template`, `eval()` and
`inference_mode()`, deterministic algorithms and greedy decoding
(`do_sample=False`, one beam). Only question/context are passed into its
English prompts. Gold answers, labels, IDs and metrics never enter the prompt.
Context is explicitly described as untrusted data, not instructions.
The input budget reserves `max_new_tokens`; only a prefix of context can be
removed. Every accepted prefix is re-templated and checked against the token
budget, preserving the complete question and instructions. Token counts,
truncation flags and input hashes are recorded.

The classifier retains question first, context second, `max_length=512` and
`truncation="only_second"`, scored in FP32. Its visible context can differ from
Qwen's because the tokenizers and budgets differ. Neither truncation policy
changes the original labels. Saved classifier context character endpoints and
generator token diagnostics make this limitation visible.

Refusal means exactly `raw_answer.strip() == 'NO_ANSWER'`, case-sensitive;
punctuation, explanations and other phrases are answer attempts. Empty output,
runtime failures and the generation limit without EOS are execution errors,
never correct abstentions. They are retained separately; a partial run is not
reported as a completed evaluation. A repaired transient failure remains in
the error log even if explicit resume later completes all pairs.

**Primary answer correctness is normalized exact match with any gold answer**,
using [standard SQuAD normalization](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py):
lowercase, remove ASCII punctuation and articles, collapse whitespace, then
take the best exact match over the gold answers. Token F1 is an additional
diagnostic metric; it never determines the primary correct/incorrect outcome.
Five successful outcomes
are exhaustive: correct/incorrect answer or unnecessary refusal for answerable
examples; correct abstention or unsupported answer for unanswerable examples.
Any attempted answer on an unanswerable example is unsupported in the benchmark
sense. EM is not a complete semantic judgement: valid paraphrases can fail EM.

For each mode, reports retain counts, class denominators and rates for all five
outcomes, plus `(correct_answer + correct_abstention) / N`. Paired differences
are filtered minus baseline in percentage points, with outcome transitions,
prevented unsupported answers and lost correct answers. H1 predicts a decrease
in unsupported-answer rate; H2 predicts an increase in overall correct outcome.
Both primary hypotheses use **automatic** outcomes, including EM-based
answerable correctness; supplementary review does not replace those results.

The paired percentile bootstrap samples **context groups** with replacement,
keeps every row of sampled groups and uses the same resampled groups for both
modes. It uses 2000 replicates, seed=42 and 95% intervals. Each replicate uses
its actual class denominators. Missing-class replicates are explicitly counted,
not replaced with zero. Intervals use valid replicates; directional support is
withheld when any replicates are invalid or fewer than two groups are available.
The two hypothesis intervals are unadjusted. This uncertainty is separate from
the sample SD across the three classifier training seeds; no pooling occurs.
Development results are exploratory, not independent final evidence for H1/H2.

### Optional supplementary human review and evidence

Human review is a separate, optional analysis. Its absence or incompleteness
does not block freeze/test. Reviewed metrics and bootstrap differences remain
supplementary and are not primary H1/H2 evidence. Automatic EM results are
retained regardless of whether review is performed.

The fixed review rubric is embedded in the protocol before test: review all
answer attempts on answerable examples; accept a context-supported, responsive
semantic equivalent of a gold answer with no contradictory/unsupported claim.
Otherwise mark it incorrect. Refusals, unanswerable attempts and execution
errors cannot be relabelled. Each decision requires ID, mode, decision and a
nonempty rationale. Identical paired answers must receive identical decisions.
No additional LLM is used as a judge.

Use development for review rehearsal; after final test, the same frozen rubric
can be applied to its predictions. The HTML export includes **all** eligible
answers, including EM matches, with the full context and original EM. Identical
raw answers within one ID appear once. Mode membership is hidden in the form
and stored in a separate `.mapping.json`; import applies the single decision
to every corresponding mode. The unchanged rubric is embedded in the form.
The export is unassessed: decision and rationale fields start empty.

```powershell
$reviewRun = "outputs/llm_comparison/$development"
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py review-export --run $reviewRun --format html --output "$reviewRun/assessment/manual_assessment.html"
# Open HTML in a browser. Manually choose correct_answer/incorrect_answer and give a rationale.
# Save draft JSON before closing; load it to continue. No automatic disk autosave.
# Export completed decisions and save the downloaded file as assessment/assessment_decisions.json.
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py review-import --run $reviewRun --decisions "$reviewRun/assessment/assessment_decisions.json" --mapping "$reviewRun/assessment/manual_assessment.mapping.json" --output-dir "$reviewRun/reviewed_results"
Get-Content "$reviewRun/reviewed_results/reviewed_metrics.json"
```

Keep the mapping unchanged and away from the assessor until decisions are made.
Only complete decisions with nonempty rationales can be imported. The package
is bound to the original source hashes, candidate set and rubric. The existing
mode/ID JSONL format remains available with `review-export --format jsonl` and
`review-import` without `--mapping`.

Raw predictions and automatic metrics remain unchanged. Reviewed metrics,
transitions and bootstrap are stored separately and clearly labelled; their
diagnostic EM/F1 values still refer to the original automatic string scoring.

Each new run is under `outputs/llm_comparison/<run_id>/`:

- `protocol.json`, `run.json`: settings/identity, purpose, status, duration,
  versions and checks. Development plans are explicitly marked unfrozen.
- `examples.jsonl`: fixed inputs and gold references for evaluation/review only.
- `classifier_scores.jsonl`: probabilities and classifier input diagnostics.
- `baseline_predictions.jsonl`, `filtered_predictions.jsonl`: raw/processed
  answers, reasons, replay identity, token diagnostics and baseline timings.
- `metrics.json`, `paired_transitions.json`, `bootstrap.json`: automatic results;
  smoke has technical metrics only and no hypothesis bootstrap.
- `errors.jsonl`: actual error outcomes by ID in either mode, referencing the
  saved examples/predictions without duplicating their full texts.
- `classifier_phase_<attempt>.json`, `generator_phase_<attempt>.json`: actual
  GPU/precision/memory and effective tokenizer/generation settings.
- `stdout.log`, `stderr.log`, `execution_errors.jsonl` when errors occur:
  progress, native/library output and failed attempts.

Predictions are flushed as each example completes. A run becomes completed
only after coverage, identity and pairing checks pass. Generated comparison
artifacts, cached weights and logs remain ignored by Git.

Suggested screenshots: classifier listing and export status; smoke phase GPU
memory report; development `show` and a saved example; frozen protocol fields;
final `show`, paired transition counts and bootstrap intervals. Screenshots of
test examples should be taken only after the final frozen run, never for tuning.

Preparation checks can be repeated without models, downloads or datasets:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' scripts/compare_llm.py --help
```

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
- The paired test is balanced by design (500 examples of each class);
  its overall rate describes that test mixture rather than natural query prevalence.
