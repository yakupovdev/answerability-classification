"""Pure scoring, pairing, identity and bootstrap utilities; no model or dataset imports."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import string
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTCOMES = ("correct_answer", "incorrect_answer", "unnecessary_refusal",
            "correct_abstention", "unsupported_answer")
METRICS = {"correct_answer_rate": ("correct_answer", "answerable"),
           "unnecessary_refusal_rate": ("unnecessary_refusal", "answerable"),
           "incorrect_answer_rate": ("incorrect_answer", "answerable"),
           "correct_abstention_rate": ("correct_abstention", "unanswerable"),
           "unsupported_answer_rate": ("unsupported_answer", "unanswerable"),
           "overall_correct_outcome": (None, "total")}
SCORING = {
    "version": 1,
    "abstention": "raw.strip() == 'NO_ANSWER', case-sensitive, no other words or punctuation",
    "invalid_output": "execution error, empty output or token limit without EOS is an execution error, never a refusal",
    "exact_match": "SQuAD: lowercase, remove ASCII punctuation, remove a/an/the, collapse whitespace; max over gold answers",
    "token_f1": "SQuAD token multiset overlap; diagnostic only",
    "unanswerable": "Any non-refusal answer is unsupported by the benchmark context, irrespective of plausibility",
    "denominators": "All planned rows of each class, including execution errors; no complete assessment until both modes cover all IDs without errors",
    "review_rubric": {
        "scope": "Review every non-refusal answer on answerable examples in both modes, without changing raw predictions",
        "correct_answer": "The answer expresses a gold answer or an equivalent meaning, supported by the provided context and responsive to the question; no contradictory or additional unsupported assertion",
        "incorrect_answer": "Wrong, incomplete, unsupported, ambiguous or non-responsive; explanations that introduce unsupported claims are incorrect",
        "fixed_outcomes": "Refusals, unanswerable attempts and execution errors cannot be relabelled by review",
        "procedure": "Human decisions with ID, mode, decision and nonempty rationale; identical paired answers require identical decisions; no LLM judge",
    },
    "hypotheses": {"H1": "filtered decreases unsupported_answer_rate",
                   "H2": "filtered increases overall_correct_outcome"},
    "inference": "Paired percentile cluster bootstrap by context_group, 2000 replicates, 95% intervals, filtered minus baseline in percentage points; unadjusted intervals for two hypotheses, no causal or general LLM claim",
}

# Analysis priority is explicit without changing the historical scoring rules or review rubric.
EVALUATION_POLICY = {
    "primary_scoring": "SQuAD normalized exact match with any gold answer",
    "primary_outcomes": "Automatic EM for answer attempts on answerable examples; existing refusal, unanswerable and execution-error rules otherwise",
    "diagnostic_scoring": ["token_f1"],
    "manual_review": {"required_for_freeze": False, "required_for_test": False,
                      "role": "optional supplementary analysis", "replaces_primary_em": False},
    "primary_hypotheses_basis": "Automatic paired test outcomes only; validation results are development diagnostics",
    "H1": "Automatic unsupported-answer rate, filtered minus baseline",
    "H2": "Automatic overall correct outcome, filtered minus baseline; answerable correctness uses normalized EM",
    "limitation": "EM can reject semantically correct paraphrases; it is not a full semantic correctness judgement",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def project_path(value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    require(path.is_relative_to(ROOT), "Paths must remain inside the project.")
    return path


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def ids_checksum(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tree_sha(directory):
    return {p.relative_to(directory).as_posix(): file_sha(p)
            for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def write_json(path, value, *, replace=False):
    import os
    import tempfile

    path = Path(path)
    require(replace or not path.exists(), f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, value):
    import os

    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path, *, recover=False):
    """Only explicit resume may repair a torn final write; keep its bytes in an audit event."""
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result, offset = [], 0
    for i, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            require(recover and i == len(lines) - 1 and not line.endswith(b"\n"),
                    f"Corrupt JSONL at {path}, line {i + 1}; manual inspection required")
            append_jsonl(path.parent / "recovery.jsonl", {
                "utc": utc_now(), "file": path.name, "discarded_tail_hex": line.hex(), "offset": offset})
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        result.append(value)
        offset += len(line)
    if lines and not lines[-1].endswith(b"\n") and len(result) == len(lines):
        require(recover, f"Unterminated JSONL record in {path}; use explicit --resume")
        with path.open("ab") as stream:
            stream.write(b"\n")
    return result


def check_ids(records, expected, *, complete=True):
    actual = [r["id"] for r in records]
    require(len(actual) == len(set(actual)), "Repeated IDs in saved records")
    require(actual == list(expected)[:len(actual)], "IDs/order differ from the saved plan")
    if complete:
        require(actual == list(expected), "Incomplete ID coverage")


def check_resume(saved, requested):
    require(digest(saved) == digest(requested),
            "Incompatible resume: data, subset, configuration, code, versions or model hashes changed. Use a new run and validate changed settings.")


def normalize_answer(text):
    text = "".join(ch for ch in text.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, golds):
    prediction = normalize_answer(prediction)
    references = [normalize_answer(gold) for gold in golds]
    references = [gold for gold in references if gold] or [""]
    exact, f1 = 0, 0.0
    for gold in references:
        exact = max(exact, int(prediction == gold))
        p, g = prediction.split(), gold.split()
        if not p or not g:
            score = float(p == g)
        else:
            common = sum((Counter(p) & Counter(g)).values())
            score = 2 * common / (len(p) + len(g))
        f1 = max(f1, score)
    return {"normalized_exact_match": exact, "token_f1": f1}


def parse_output(raw, *, limit_reached=False):
    require(isinstance(raw, str), "Raw output must be text")
    processed = raw.strip()
    error = "generation_limit" if limit_reached else "empty_output" if not processed else None
    return {"raw_answer": raw, "processed_answer": processed,
            "status": "error" if error else "ok", "error": error,
            "abstained": not error and processed == "NO_ANSWER",
            "reason": "generator_no_answer" if not error and processed == "NO_ANSWER" else error}


def gate_replay(baseline, probability, threshold):
    require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid classifier probability")
    require(math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid gate threshold")
    if probability >= threshold:
        return {**baseline, "mode": "filtered", "paired_replay": True,
                "p_answerable": probability, "threshold": threshold}
    return {"id": baseline["id"], "mode": "filtered", "raw_answer": None,
            "processed_answer": "NO_ANSWER", "status": "ok", "abstained": True,
            "reason": "classifier_gate", "error": None, "paired_replay": True,
            "p_answerable": probability, "threshold": threshold,
            "generator_identity": baseline["generator_identity"],
            "input_sha256": baseline["input_sha256"]}


def score_row(example, prediction, review=None):
    if prediction["status"] != "ok":
        return {"outcome": "execution_error", "normalized_exact_match": 0, "token_f1": 0.0}
    if prediction["abstained"]:
        correct = not example["label"]
        return {"outcome": "correct_abstention" if correct else "unnecessary_refusal",
                "normalized_exact_match": int(correct), "token_f1": float(correct)}
    scores = answer_scores(prediction["processed_answer"], example["answers"]["text"]) if example["label"] else {
        "normalized_exact_match": 0, "token_f1": 0.0}
    outcome = ("correct_answer" if scores["normalized_exact_match"] else "incorrect_answer") if example["label"] else "unsupported_answer"
    if review is not None:
        require(example["label"] == 1 and review in ("correct_answer", "incorrect_answer"), "Review cannot alter this outcome")
        outcome = review
    return {"outcome": outcome, **scores}


def mode_metrics(examples, predictions, reviews=None):
    check_ids(predictions, [e["id"] for e in examples])
    counts = Counter({key: 0 for key in (*OUTCOMES, "execution_error")})
    sums = Counter()
    for e, p in zip(examples, predictions):
        score = score_row(e, p, (reviews or {}).get((p["mode"], e["id"])))
        counts[score["outcome"]] += 1
        for key in ("normalized_exact_match", "token_f1"):
            sums[key] += score[key]
    denominators = {"total": len(examples), "answerable": sum(e["label"] == 1 for e in examples),
                    "unanswerable": sum(e["label"] == 0 for e in examples)}
    rates = {}
    for key, (category, denom) in METRICS.items():
        numerator = counts[category] if category else counts["correct_answer"] + counts["correct_abstention"]
        n = denominators[denom]
        rates[key] = {"count": numerator, "denominator": n, "rate": numerator / n if n else None}
    return {"counts": dict(counts), "denominators": denominators, "rates": rates,
            "complete_without_errors": counts["execution_error"] == 0,
            "diagnostics": {key: {"sum": value, "denominator": len(examples),
                                  "mean": value / len(examples) if examples else None} for key, value in sums.items()}}


def paired_report(examples, baseline, filtered, reviews=None):
    ids = [e["id"] for e in examples]
    check_ids(baseline, ids)
    check_ids(filtered, ids)
    before, after = mode_metrics(examples, baseline, reviews), mode_metrics(examples, filtered, reviews)
    transitions, prevented, lost = Counter(), 0, 0
    for e, b, f in zip(examples, baseline, filtered):
        left = score_row(e, b, (reviews or {}).get(("baseline", e["id"])))["outcome"]
        right = score_row(e, f, (reviews or {}).get(("filtered", e["id"])))["outcome"]
        transitions[(left, right)] += 1
        prevented += int(left == "unsupported_answer" and right == "correct_abstention")
        lost += int(left == "correct_answer" and right != "correct_answer")
    changes = {}
    for key in METRICS:
        b, f = before["rates"][key]["rate"], after["rates"][key]["rate"]
        changes[key] = None if b is None or f is None else (f - b) * 100
    return {"baseline": before, "filtered": after, "filtered_minus_baseline_pp": changes,
            "transitions": [{"baseline": b, "filtered": f, "count": n} for (b, f), n in sorted(transitions.items())],
            "prevented_unsupported_answers": prevented, "lost_correct_answers": lost,
            "reviewed": reviews is not None, "paired_replay": True,
            "analysis_role": "optional_supplementary_review" if reviews is not None else "primary_automatic_em",
            "answer_correctness_basis": "supplementary_manual_review" if reviews is not None else "normalized_exact_match_any_gold",
            "evaluation_policy": EVALUATION_POLICY,
            "timing_note": "Replay is not a measured gated-inference speedup; no filtered inference latency is claimed"}


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    left = int(index)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (index - left)


def cluster_bootstrap(examples, baseline, filtered, *, repeats=2000, seed=42, reviews=None):
    report = paired_report(examples, baseline, filtered, reviews)
    require(report["baseline"]["complete_without_errors"] and report["filtered"]["complete_without_errors"],
            "Bootstrap requires complete pairs without execution errors")
    # Cluster sufficient statistics preserve every question and each replicate's class denominators.
    groups = defaultdict(Counter)
    for e, b, f in zip(examples, baseline, filtered):
        group = groups[e["context_group"]]
        group["total"] += 1
        group["answerable" if e["label"] else "unanswerable"] += 1
        for mode, prediction in (("baseline", b), ("filtered", f)):
            category = score_row(e, prediction, (reviews or {}).get((mode, e["id"])))["outcome"]
            group[f"{mode}:{category}"] += 1
    require(groups and repeats > 0, "Need nonempty groups and positive bootstrap repeats")
    clusters = [groups[key] for key in sorted(groups)]
    rng = random.Random(seed)
    values = {key: [] for key in METRICS}
    for _ in range(repeats):
        aggregate = Counter()
        for _ in clusters:
            aggregate.update(clusters[rng.randrange(len(clusters))])
        for key, (category, denom) in METRICS.items():
            if aggregate[denom] == 0:
                continue
            def numerator(mode):
                return aggregate[f"{mode}:{category}"] if category else (
                    aggregate[f"{mode}:correct_answer"] + aggregate[f"{mode}:correct_abstention"])
            values[key].append(100 * (numerator("filtered") - numerator("baseline")) / aggregate[denom])
    differences = {key: {"estimate_pp": report["filtered_minus_baseline_pp"][key],
                         "ci95_pp": [percentile(v, .025), percentile(v, .975)] if v else None,
                         "valid_replicates": len(v), "missing_class_replicates": repeats - len(v)}
                   for key, v in values.items()}
    hypotheses = {}
    primary_hypotheses = (("H1", "unsupported_answer_rate", -1), ("H2", "overall_correct_outcome", 1)) if reviews is None else ()
    for hypothesis, metric, sign in primary_hypotheses:
        result = differences[metric]
        ci = result["ci95_pp"]
        usable = ci is not None and result["valid_replicates"] == repeats and len(clusters) > 1
        hypotheses[hypothesis] = {"metric": metric, "expected_direction": "decrease" if sign == -1 else "increase",
                                  "direction_supported_by_unadjusted_ci": (ci[1] < 0 if sign == -1 else ci[0] > 0) if usable else None}
    return {"unit": "context_group", "paired": True, "repeats": repeats, "seed": seed,
            "unique_groups": len(clusters), "confidence": 0.95, "method": "percentile; unadjusted",
            "differences": differences, "hypotheses": hypotheses,
            "analysis_role": report["analysis_role"], "answer_correctness_basis": report["answer_correctness_basis"],
            "evaluation_policy": EVALUATION_POLICY,
            "missing_class_policy": "Intervals use valid replicates only; missing-class counts are reported, never substituted by zero. Directional support is withheld if any replicates are invalid or fewer than two groups are available.",
            "interpretation": "Clustered validation/test sampling uncertainty, not classifier training-seed sample SD; no pooling of classifier seeds"}


def review_candidates(examples, baseline, filtered):
    ids = [e["id"] for e in examples]
    check_ids(baseline, ids)
    check_ids(filtered, ids)
    rows = []
    for e, b, f in zip(examples, baseline, filtered):
        for p in (b, f):
            if e["label"] == 1 and p["status"] == "ok" and not p["abstained"]:
                rows.append({"id": e["id"], "mode": p["mode"], "question": e["question"], "context": e["context"],
                             "gold_answers": e["answers"]["text"], "raw_answer": p["raw_answer"],
                             "generator_input_diagnostics": p.get("input"),
                             "automatic_outcome": score_row(e, p)["outcome"],
                             "original_em": score_row(e, p)["normalized_exact_match"], "decision": "", "rationale": ""})
    return rows


def validate_reviews(candidates, decisions):
    expected = {(r["mode"], r["id"]): r for r in candidates}
    result, paired = {}, {}
    for row in decisions:
        key = (row["mode"], row["id"])
        require(key in expected and key not in result, "Unknown or duplicate review ID/mode")
        require(row["decision"] in ("correct_answer", "incorrect_answer") and isinstance(row["rationale"], str)
                and row["rationale"].strip(), "Each review requires a permitted decision and rationale")
        result[key] = row["decision"]
        pair = (row["id"], expected[key]["raw_answer"])
        require(pair not in paired or paired[pair] == row["decision"], "Identical paired answers received inconsistent decisions")
        paired[pair] = row["decision"]
    require(set(result) == set(expected), "Review is incomplete; review every exported answer before producing reviewed metrics")
    return result


def review_bundle(candidates, rubric, source):
    """Deduplicate exact raw answers within an ID; keep mode membership outside the form."""
    cards, membership, seen = [], [], {}
    for row in candidates:
        key = (row["id"], row["raw_answer"])
        if key not in seen:
            review_id = digest(list(key))
            seen[key] = len(cards)
            cards.append({"review_id": review_id, **{field: row[field] for field in
                          ("id", "question", "context", "gold_answers", "raw_answer", "original_em")},
                          "decision": "", "rationale": ""})
            membership.append({"review_id": review_id, "id": row["id"],
                               "raw_answer_sha256": digest(row["raw_answer"]), "modes": []})
        index = seen[key]
        require(all(cards[index][field] == row[field] for field in
                    ("question", "context", "gold_answers", "original_em")), "Identical ID/answer has inconsistent reference fields")
        require(row["mode"] not in membership[index]["modes"], "Repeated candidate ID/mode")
        membership[index]["modes"].append(row["mode"])
    binding = {"schema": "answer-assessment-package-v1", "source": source,
               "rubric": rubric, "cards": cards, "membership": membership}
    package_sha256 = digest(binding)
    form = {"schema": binding["schema"], "package_sha256": package_sha256,
            "rubric": rubric, "cards": cards}
    mapping = {key: value for key, value in binding.items() if key != "cards"}
    mapping["package_sha256"] = package_sha256
    return form, mapping


def expand_review_bundle(candidates, rubric, source, mapping, decisions):
    """Bind human decisions to the exact source predictions, then expand to existing mode/ID validation."""
    form, expected = review_bundle(candidates, rubric, source)
    require(mapping == expected, "Assessment mapping differs from the current source run, rubric or full candidate set")
    require(decisions.get("schema") == "answer-assessment-decisions-v1"
            and decisions.get("package_sha256") == form["package_sha256"], "Decisions belong to a different assessment package")
    entries = {item["review_id"]: item for item in mapping["membership"]}
    seen, expanded = set(), []
    for decision in decisions["decisions"]:
        key = decision["review_id"]
        require(key in entries and key not in seen, "Unknown or duplicate assessment review_id")
        seen.add(key)
        entry = entries[key]
        require(decision["id"] == entry["id"], "Assessment ID does not match mapping")
        expanded.extend({"id": entry["id"], "mode": mode, "decision": decision["decision"],
                         "rationale": decision["rationale"]} for mode in entry["modes"])
    require(seen == set(entries), "Assessment is incomplete; every unique ID/answer needs a human decision")
    return validate_reviews(candidates, expanded), expanded


def assessment_html(form):
    """Offline form: no external resources or mode membership, no automatic decisions."""
    payload = json.dumps(form, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Answer assessment — awaiting decisions</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;color:#19232d;background:#f2f4f6}
main{max-width:1050px;margin:auto;padding:24px}header,.card,details.rubric{background:white;border:1px solid #ccd3da;border-radius:8px;padding:20px;margin:14px 0}
.bar{position:sticky;top:0;background:#f2f4f6;padding:12px 0;border-bottom:1px solid #ccd3da;z-index:1}
button,select,input,textarea{font:inherit;padding:8px}button{cursor:pointer;margin:4px}button:disabled{cursor:default}
textarea{box-sizing:border-box;width:100%;min-height:95px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#f7f8fa;padding:12px;border-radius:4px}
.context{border-left:3px solid #91a5b5}h2{font-size:20px}h3{font-size:16px;margin-bottom:4px}label{display:block;margin:8px 0}.muted{color:#4b5b69}.complete{border-left:5px solid #24704c}.notice{color:#7b4200}
</style><main><header><h1>Answer assessment</h1>
<p class="notice">Unassessed export. No human decisions have been supplied. This form includes all eligible answers, including exact matches.</p>
<p>Assess each answer using the unchanged rubric. Mode membership is kept in a separate mapping file. The original EM is available in each card; it is not a suggested decision.</p>
<p>Choose <b>correct_answer</b> or <b>incorrect_answer</b> and explain why. Save a draft before closing this page; changes are kept in memory only. Load that draft to continue. Completed decisions are downloaded as JSON for CLI import. This page sends nothing over the network.</p></header>
<details class="rubric" open><summary><b>Original review rubric (unchanged)</b></summary><div id="rubric"></div></details>
<div class="bar"><strong id="progress"></strong><br>
<button id="draft" type="button">Save draft JSON</button><button id="export" type="button" disabled>Export completed decisions</button>
<label style="display:inline-block">Load draft/decisions <input id="load" type="file" accept=".json,application/json"></label>
<label>Display <select id="filter"><option value="all">All cards</option><option value="pending">Cards still requiring a decision/rationale</option></select></label>
<span id="message" role="status"></span></div><div id="cards"></div></main>
<script id="assessment-data" type="application/json">''' + payload + '''</script>
<script>
'use strict';
const form=JSON.parse(document.getElementById('assessment-data').textContent);
const cards=form.cards; const views=[];
function node(tag,text,parent){const el=document.createElement(tag);if(text!==undefined)el.textContent=text;if(parent)parent.appendChild(el);return el;}
function done(c){return ['correct_answer','incorrect_answer'].includes(c.decision)&&c.rationale.trim().length>0;}
function update(){const n=cards.filter(done).length;document.getElementById('progress').textContent=n+' / '+cards.length+' complete';document.getElementById('export').disabled=n!==cards.length;views.forEach((v,i)=>{v.classList.toggle('complete',done(cards[i]));v.hidden=document.getElementById('filter').value==='pending'&&done(cards[i])&&!v.contains(document.activeElement);});}
Object.entries(form.rubric).forEach(([key,value])=>{node('h3',key,document.getElementById('rubric'));node('p',value,document.getElementById('rubric'));});
cards.forEach((c,i)=>{
 const card=node('section',undefined,document.getElementById('cards'));card.className='card';views.push(card);
 node('h2','Card '+(i+1)+' — ID: '+c.id,card);
 node('h3','Question',card);node('pre',c.question,card);
 node('h3','Full original context',card);node('pre',c.context,card).className='context';
 node('h3','Gold answers',card);node('pre',JSON.stringify(c.gold_answers,null,2),card);
 node('h3','Raw model answer (unchanged)',card);node('pre',c.raw_answer,card);
 const em=node('details',undefined,card);node('summary','Original automatic EM (click to show)',em);node('p',String(c.original_em),em);
 const label=node('label','Human decision: ',card);const select=node('select',undefined,label);
 [['','Choose a decision'],['correct_answer','correct_answer'],['incorrect_answer','incorrect_answer']].forEach(([value,text])=>{const option=node('option',text,select);option.value=value;});
 const rationaleLabel=node('label','Rationale (required):',card);const rationale=node('textarea',undefined,rationaleLabel);
 select.addEventListener('change',()=>{c.decision=select.value;update();});rationale.addEventListener('input',()=>{c.rationale=rationale.value;update();});
 card._select=select;card._rationale=rationale;
});
function download(completed){if(completed&&!cards.every(done))throw Error('Complete every card first');const result={schema:'answer-assessment-decisions-v1',package_sha256:form.package_sha256,decisions:cards.map(c=>({review_id:c.review_id,id:c.id,decision:c.decision,rationale:c.rationale}))};const blob=new Blob([JSON.stringify(result,null,2)+'\\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const link=document.createElement('a');link.href=url;link.download=completed?'assessment_decisions.json':'assessment_draft.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.getElementById('draft').onclick=()=>download(false);document.getElementById('export').onclick=()=>download(true);
document.getElementById('filter').onchange=update;
document.addEventListener('focusin',update);
document.getElementById('load').onchange=async event=>{try{const file=event.target.files[0];if(!file)return;const saved=JSON.parse(await file.text());if(saved.schema!=='answer-assessment-decisions-v1'||saved.package_sha256!==form.package_sha256)throw Error('Wrong assessment package');if(!Array.isArray(saved.decisions)||saved.decisions.length!==cards.length)throw Error('Wrong number of cards');const byId=new Map();saved.decisions.forEach(d=>{if(byId.has(d.review_id))throw Error('Duplicate review_id');byId.set(d.review_id,d);});const restored=cards.map(c=>{const d=byId.get(c.review_id);if(!d||d.id!==c.id||!['','correct_answer','incorrect_answer'].includes(d.decision)||typeof d.rationale!=='string')throw Error('Invalid decision/ID');return d;});restored.forEach((d,i)=>{cards[i].decision=d.decision;cards[i].rationale=d.rationale;views[i]._select.value=d.decision;views[i]._rationale.value=d.rationale;});update();document.getElementById('message').textContent='Loaded saved progress';}catch(error){document.getElementById('message').textContent=error.message;}};
update();
</script></html>
'''
