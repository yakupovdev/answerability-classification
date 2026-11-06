"""Synthetic methodological checks; never open project datasets or model weights."""

import copy
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from comparison_utils import (SCORING, EVALUATION_POLICY, answer_scores, check_ids, check_resume, cluster_bootstrap,
                              digest, gate_replay, ids_checksum, mode_metrics, normalize_answer,
                              paired_report, parse_output, read_jsonl, review_candidates,
                              validate_reviews, write_json, review_bundle, expand_review_bundle,
                              assessment_html)
from compare_llm import (check_frozen_runtime, classifier_input, ensure_filtered,
                         execute_run, freeze_protocol, parser, prompt_input, records_for_run, save_analysis)


def example(identifier, label, group, answer="Paris"):
    return {"id": identifier, "label": label, "context_group": group,
            "question": "Which city?", "context": "The city is Paris.",
            "answers": {"text": [answer] if label else [], "answer_start": [12] if label else []}}


def prediction(identifier, text, identity="generator", **kwargs):
    return {"id": identifier, "mode": "baseline", "generator_identity": identity,
            "input_sha256": "tokens", **parse_output(text, **kwargs)}


class ScoringTests(unittest.TestCase):
    def test_squad_normalization_and_multiple_golds(self):
        self.assertEqual(normalize_answer(" The, Eiffel Tower! "), "eiffel tower")
        self.assertEqual(answer_scores("Paris.", ["Lyon", "the Paris"])["normalized_exact_match"], 1)
        self.assertAlmostEqual(answer_scores("red blue", ["red green"])["token_f1"], .5)
        self.assertAlmostEqual(answer_scores("red red", ["red"])["token_f1"], 2 / 3)

    def test_strict_refusal_and_invalid_generation(self):
        self.assertTrue(parse_output(" \nNO_ANSWER\n")["abstained"])
        for text in ("no", "no_answer", "NO_ANSWER.", "There is no answer", "NO_ANSWER because unknown"):
            self.assertFalse(parse_output(text)["abstained"])
        for value in (parse_output("  "), parse_output("NO_ANSWER", limit_reached=True)):
            self.assertEqual(value["status"], "error")
            self.assertFalse(value["abstained"])

    def test_all_five_outcomes_and_denominators(self):
        rows = [example(str(i), int(i < 3), str(i)) for i in range(5)]
        predictions = [prediction(str(i), text) for i, text in enumerate(("Paris", "Lyon", "NO_ANSWER", "NO_ANSWER", "Maybe"))]
        result = mode_metrics(rows, predictions)
        self.assertEqual(set(result["counts"].values()), {0, 1})
        self.assertEqual(result["rates"]["correct_answer_rate"], {"count": 1, "denominator": 3, "rate": 1 / 3})
        self.assertEqual(result["rates"]["unsupported_answer_rate"]["rate"], .5)
        self.assertEqual(result["rates"]["overall_correct_outcome"]["rate"], .4)
        predictions[0] = prediction("0", "", limit_reached=True)
        result = mode_metrics(rows, predictions)
        self.assertEqual(result["counts"]["execution_error"], 1)
        self.assertEqual(result["rates"]["correct_answer_rate"]["denominator"], 3)
        self.assertFalse(result["complete_without_errors"])

    def test_missing_class_is_explicit(self):
        result = mode_metrics([example("a", 1, "g")], [prediction("a", "Paris")])
        self.assertIsNone(result["rates"]["unsupported_answer_rate"]["rate"])
        self.assertEqual(result["rates"]["unsupported_answer_rate"]["denominator"], 0)


class PairingTests(unittest.TestCase):
    def test_gate_boundary_and_shared_answer(self):
        base = prediction("a", "Paris")
        before = copy.deepcopy(base)
        self.assertEqual(gate_replay(base, .5, .5)["raw_answer"], "Paris")
        gate = gate_replay(base, .499, .5)
        self.assertTrue(gate["abstained"])
        self.assertIsNone(gate["raw_answer"])
        self.assertEqual(gate["reason"], "classifier_gate")
        self.assertEqual(base, before)
        for p in (-.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                gate_replay(base, p, .5)

    def test_paired_changes_transitions_and_losses(self):
        rows = [example("a", 1, "g"), example("b", 0, "g")]
        base = [prediction("a", "Paris"), prediction("b", "Lyon")]
        filtered = [gate_replay(p, .1, .5) for p in base]
        result = paired_report(rows, base, filtered)
        self.assertEqual(result["prevented_unsupported_answers"], 1)
        self.assertEqual(result["lost_correct_answers"], 1)
        self.assertEqual(result["filtered_minus_baseline_pp"]["unsupported_answer_rate"], -100)
        self.assertEqual(result["filtered_minus_baseline_pp"]["overall_correct_outcome"], 0)
        self.assertEqual(sum(row["count"] for row in result["transitions"]), 2)

    def test_id_coverage_order_and_duplicates(self):
        check_ids([{"id": "a"}], ["a", "b"], complete=False)
        for records in ([{"id": "b"}, {"id": "a"}], [{"id": "a"}, {"id": "a"}], [{"id": "a"}]):
            with self.assertRaises(ValueError):
                check_ids(records, ["a", "b"])

    def test_resume_rejects_config_model_data_version_and_order_changes(self):
        saved = {"config": {"dtype": "bfloat16"}, "model": "sha", "ids": ["a", "b"], "version": "1"}
        check_resume(saved, copy.deepcopy(saved))
        for key, value in (("config", {"dtype": "float16"}), ("model", "other"), ("ids", ["b", "a"]), ("version", "2")):
            altered = {**saved, key: value}
            with self.assertRaisesRegex(ValueError, "Incompatible resume"):
                check_resume(saved, altered)

    def test_final_runtime_must_match_validated_protocol(self):
        actual = {"gpu": "synthetic GPU", "precision": "bfloat16", "effective_generation_config": {"do_sample": False}}
        meta = {"identity": {"protocol": {"path": "outputs/synthetic_protocol.json", "sha256": "sha"}}}
        with patch("compare_llm.file_sha", return_value="sha"), \
                patch("compare_llm.read_json", return_value={"generator_runtime": actual}):
            check_frozen_runtime(meta, "generator", actual)
            with self.assertRaisesRegex(ValueError, "differ from validated protocol"):
                check_frozen_runtime(meta, "generator", {**actual, "precision": "float16"})

    def test_replay_rejects_changed_generator_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                ensure_filtered(Path(folder), [example("a", 1, "g")], [prediction("a", "Paris")],
                                [{"id": "a", "p_answerable": .8}], {"different": "settings"}, .5)

    def test_atomic_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            write_json(path, {"a": 1})
            with self.assertRaises(ValueError):
                write_json(path, {"a": 2})
            self.assertEqual(json.loads(path.read_text()), {"a": 1})

    def test_torn_jsonl_repair_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.jsonl"
            path.write_bytes(b'{"id":"a"}\n{"id":')
            with self.assertRaises(ValueError):
                read_jsonl(path)
            self.assertEqual(read_jsonl(path, recover=True), [{"id": "a"}])
            self.assertTrue((path.parent / "recovery.jsonl").is_file())
            self.assertEqual(path.read_bytes(), b'{"id":"a"}\n')

    def test_final_coverage_requires_all_1000_ids(self):
        settings = {"classifier": {}, "data": {"reference": {"splits": {"test": {"ids_sha256": ids_checksum(["a"])}}}}}
        row = example("a", 1, "group")
        base = prediction("a", "Paris", digest(settings))
        score = {"id": "a", "p_answerable": .8, "classifier_identity": digest({})}
        meta = {"purpose": "test", "identity": {"settings": settings, "threshold": .5,
                "plan": {"ids": ["a"], "content_sha256": digest([row])}}}
        records = {"examples.jsonl": [row], "baseline_predictions.jsonl": [base],
                   "filtered_predictions.jsonl": [gate_replay(base, .8, .5)], "classifier_scores.jsonl": [score]}
        with patch("compare_llm.read_json", return_value=meta), patch("compare_llm.read_jsonl", side_effect=lambda path: records[path.name]):
            with self.assertRaisesRegex(ValueError, "1000"):
                records_for_run(Path("synthetic"))


class BootstrapTests(unittest.TestCase):
    def test_paired_clusters_keep_all_rows_and_denominators(self):
        # Each group contains both classes and two questions; paired effects are constant.
        rows = [example(str(i), i % 2, str(i // 2)) for i in range(8)]
        base = [prediction(r["id"], "Paris" if r["label"] else "Lyon") for r in rows]
        filtered = [gate_replay(b, .1, .5) for b in base]
        first = cluster_bootstrap(rows, base, filtered, repeats=2000, seed=42)
        self.assertEqual(first, cluster_bootstrap(rows, base, filtered, repeats=2000, seed=42))
        self.assertEqual(first["unique_groups"], 4)
        self.assertEqual(first["differences"]["unsupported_answer_rate"]["ci95_pp"], [-100, -100])
        self.assertEqual(first["differences"]["overall_correct_outcome"]["ci95_pp"], [0, 0])
        self.assertEqual(first["differences"]["unsupported_answer_rate"]["valid_replicates"], 2000)

    def test_zero_class_replicates_are_not_silently_zero(self):
        rows = [example("a", 1, "g1"), example("b", 0, "g2")]
        base = [prediction("a", "Paris"), prediction("b", "Lyon")]
        filtered = [gate_replay(b, .1, .5) for b in base]
        result = cluster_bootstrap(rows, base, filtered, repeats=200, seed=42)
        rate = result["differences"]["unsupported_answer_rate"]
        self.assertGreater(rate["missing_class_replicates"], 0)
        self.assertEqual(rate["valid_replicates"] + rate["missing_class_replicates"], 200)
        self.assertEqual(rate["ci95_pp"], [-100, -100])

    def test_cluster_weighting_not_equal_weight_per_question_or_group_rate(self):
        rows = [example("a", 0, "small")] + [example(str(i), 0, "large") for i in range(3)]
        base = [prediction(r["id"], "wrong") for r in rows]
        filtered = [gate_replay(b, 0 if i == 0 else 1, .5) for i, b in enumerate(base)]
        result = cluster_bootstrap(rows, base, filtered, repeats=500, seed=42)
        rate = result["differences"]["unsupported_answer_rate"]
        self.assertEqual(rate["estimate_pp"], -25)
        self.assertEqual(rate["ci95_pp"], [-100, 0])


class ReviewTests(unittest.TestCase):
    def test_semantic_review_separate_from_em_and_paired_consistency(self):
        rows = [example("a", 1, "g", "New York City")]
        b = [prediction("a", "NYC")]
        f = [gate_replay(b[0], .8, .5)]
        candidates = review_candidates(rows, b, f)
        self.assertEqual(len(candidates), 2)
        decisions = [{"id": "a", "mode": mode, "decision": "correct_answer", "rationale": "Equivalent city name supported by context"}
                     for mode in ("baseline", "filtered")]
        reviews = validate_reviews(candidates, decisions)
        self.assertEqual(paired_report(rows, b, f)["baseline"]["counts"]["correct_answer"], 0)
        self.assertEqual(paired_report(rows, b, f, reviews)["baseline"]["counts"]["correct_answer"], 1)
        self.assertEqual(b[0]["raw_answer"], "NYC")
        decisions[1]["decision"] = "incorrect_answer"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_reviews(candidates, decisions)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_reviews(candidates, [])

    def test_blind_bundle_includes_em_matches_and_expands_identical_pairs(self):
        rows = [example("a", 1, "g1"), example("b", 1, "g2", "New York City"), example("c", 0, "g3")]
        b = [prediction("a", "Paris"), prediction("b", "NYC"), prediction("c", "Lyon")]
        f = [gate_replay(p, .8 if i == 0 else .1, .5) for i, p in enumerate(b)]
        candidates = review_candidates(rows, b, f)
        form, mapping = review_bundle(candidates, SCORING["review_rubric"], {"source": "synthetic"})
        self.assertEqual(len(candidates), 3)
        self.assertEqual(len(form["cards"]), 2)
        self.assertEqual([card["original_em"] for card in form["cards"]], [1, 0])
        self.assertEqual(mapping["membership"][0]["modes"], ["baseline", "filtered"])
        self.assertNotIn('"modes"', json.dumps(form))
        self.assertTrue(all(c["decision"] == c["rationale"] == "" for c in form["cards"]))
        self.assertEqual(form["rubric"], SCORING["review_rubric"])
        decisions = {"schema": "answer-assessment-decisions-v1", "package_sha256": form["package_sha256"],
                     "decisions": [{"review_id": c["review_id"], "id": c["id"], "decision": "correct_answer",
                                    "rationale": "Synthetic human decision fixture"} for c in form["cards"]]}
        reviews, expanded = expand_review_bundle(candidates, form["rubric"], {"source": "synthetic"}, mapping, decisions)
        self.assertEqual(len(expanded), 3)
        self.assertEqual(reviews[("baseline", "a")], reviews[("filtered", "a")])
        automatic = paired_report(rows, b, f)
        reviewed = paired_report(rows, b, f, reviews)
        self.assertEqual(automatic["baseline"]["counts"]["correct_answer"], 1)
        self.assertEqual(reviewed["baseline"]["counts"]["correct_answer"], 2)
        self.assertFalse(automatic["reviewed"])
        self.assertTrue(reviewed["reviewed"])

    def test_bundle_rejects_blank_incomplete_duplicate_changed_source_and_mapping(self):
        row = example("a", 1, "g")
        b = [prediction("a", "Paris")]
        candidates = review_candidates([row], b, [gate_replay(b[0], .8, .5)])
        rubric, source = SCORING["review_rubric"], {"source": "synthetic"}
        form, mapping = review_bundle(candidates, rubric, source)
        decision = {"review_id": form["cards"][0]["review_id"], "id": "a", "decision": "correct_answer", "rationale": "Reason"}
        doc = {"schema": "answer-assessment-decisions-v1", "package_sha256": form["package_sha256"], "decisions": [decision]}
        for changed in ({**doc, "decisions": []}, {**doc, "decisions": [decision, decision]},
                        {**doc, "decisions": [{**decision, "decision": ""}]},
                        {**doc, "decisions": [{**decision, "rationale": " "}]},
                        {**doc, "decisions": [{**decision, "id": "wrong"}]},
                        {**doc, "package_sha256": "wrong"}):
            with self.assertRaises(ValueError):
                expand_review_bundle(candidates, rubric, source, mapping, changed)
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            expand_review_bundle(candidates, rubric, {"source": "changed"}, mapping, doc)
        tampered = copy.deepcopy(mapping)
        tampered["membership"][0]["modes"] = ["baseline"]
        with self.assertRaisesRegex(ValueError, "mapping differs"):
            expand_review_bundle(candidates, rubric, source, tampered, doc)

    def test_deduplication_is_only_exact_raw_answer_within_id(self):
        rows = [example("a", 1, "g1"), example("b", 1, "g2")]
        b = [prediction(r["id"], "Paris") for r in rows]
        f = [{**prediction(r["id"], "Paris." if i == 0 else "Paris"), "mode": "filtered"} for i, r in enumerate(rows)]
        form, _ = review_bundle(review_candidates(rows, b, f), SCORING["review_rubric"], {})
        self.assertEqual(len(form["cards"]), 3)
        self.assertEqual(len({c["review_id"] for c in form["cards"]}), 3)

    def test_html_escapes_context_and_contains_no_mode_assignments(self):
        rows = [example("a", 1, "g")]
        rows[0]["context"] = '</script><script>unsafe()</script> & full context'
        b = [prediction("a", "Paris")]
        form, _ = review_bundle(review_candidates(rows, b, [gate_replay(b[0], .8, .5)]), SCORING["review_rubric"], {})
        html = assessment_html(form)
        self.assertNotIn('<script>unsafe()', html)
        self.assertNotIn('"modes"', html)
        payload = html.split('<script id="assessment-data" type="application/json">')[1].split('</script>')[0]
        self.assertEqual(json.loads(payload), form)
        self.assertEqual(json.loads(payload)["cards"][0]["context"], rows[0]["context"])


class EvaluationPolicyTests(unittest.TestCase):
    def test_primary_em_is_not_replaced_by_token_f1_or_supplementary_review(self):
        rows = [example("a", 1, "g1", "red green"), example("b", 0, "g2")]
        b = [prediction("a", "red blue"), prediction("b", "Lyon")]
        f = [gate_replay(p, .1, .5) for p in b]
        primary = paired_report(rows, b, f)
        self.assertEqual(primary["baseline"]["counts"]["incorrect_answer"], 1)
        self.assertEqual(primary["baseline"]["diagnostics"]["token_f1"]["sum"], .5)
        reviews = {("baseline", "a"): "correct_answer"}
        supplementary = paired_report(rows, b, f, reviews)
        self.assertEqual(supplementary["baseline"]["counts"]["correct_answer"], 1)
        self.assertEqual(supplementary["analysis_role"], "optional_supplementary_review")
        self.assertEqual(supplementary["answer_correctness_basis"], "supplementary_manual_review")
        self.assertEqual(primary["answer_correctness_basis"], "normalized_exact_match_any_gold")
        self.assertEqual(primary, paired_report(rows, b, f))
        automatic_bootstrap = cluster_bootstrap(rows, b, f, repeats=30)
        reviewed_bootstrap = cluster_bootstrap(rows, b, f, repeats=30, reviews=reviews)
        self.assertEqual(set(automatic_bootstrap["hypotheses"]), {"H1", "H2"})
        self.assertEqual(reviewed_bootstrap["hypotheses"], {})
        self.assertFalse(EVALUATION_POLICY["manual_review"]["replaces_primary_em"])

    def test_supplementary_analysis_cannot_use_primary_output_names(self):
        with self.assertRaisesRegex(ValueError, "primary automatic results must remain separate"):
            save_analysis(Path("synthetic"), {}, [], [], [], reviews={})

    def test_frozen_protocol_records_primary_policy_without_review_files(self):
        # Only synthetic objects and mocked I/O: no actual freeze command/protocol or datasets.
        settings, plan = {"synthetic": True}, {"ids": ["a", "b"]}
        meta = {"identity": {"settings": settings, "plan": plan}}
        rows = [example("a", 1, "g1"), example("b", 0, "g2")]
        b = [prediction("a", "Paris"), prediction("b", "Lyon")]
        f = [gate_replay(p, .1, .5) for p in b]
        args = argparse.Namespace(development_run="outputs/synthetic-development", smoke_run="outputs/synthetic-smoke",
                                  config="configs/llm_comparison.json", classifier_run="outputs/synthetic-classifier",
                                  classifier_model="models/synthetic", generator_lock="outputs/synthetic-lock.json",
                                  threshold=.5, threshold_report=None, output="outputs/synthetic-protocol.json")
        phase = {"gpu": "synthetic", "total_bytes": 1, "cuda_build": "synthetic", "precision": "synthetic",
                 "deterministic_algorithms": True, "tf32": False, "inference_seed": 42,
                 "tokenizer": {}, "effective_generation_config": {"do_sample": False}}
        with patch("compare_llm.verify_completed", return_value=(meta, rows, b, f)), \
                patch("compare_llm.descriptor", return_value=settings), \
                patch("compare_llm.validation_plan", return_value=(rows, plan)), \
                patch("compare_llm.latest_phase", return_value=phase), \
                patch("compare_llm.file_sha", return_value="synthetic"), \
                patch("compare_llm.review_candidates", side_effect=AssertionError("Review must not be required")), \
                patch("compare_llm.write_json") as write, contextlib.redirect_stdout(io.StringIO()):
            freeze_protocol(args)
        protocol = write.call_args.args[1]
        self.assertEqual(protocol["evaluation_policy"], EVALUATION_POLICY)
        self.assertFalse(protocol["evaluation_policy"]["manual_review"]["required_for_freeze"])
        self.assertFalse(protocol["evaluation_policy"]["manual_review"]["required_for_test"])
        self.assertEqual(protocol["review_rubric"], SCORING["review_rubric"])
        self.assertEqual(protocol["protocol_sha256"], digest({k: v for k, v in protocol.items() if k != "protocol_sha256"}))

    def test_test_requires_explicit_primary_policy_before_opening_any_data(self):
        args = argparse.Namespace(command="test", protocol="outputs/synthetic-protocol.json")
        protocol = {"schema": "paired-llm-protocol-v1", "frozen": True}
        protocol["protocol_sha256"] = digest(protocol)
        with patch("compare_llm.read_json", return_value=protocol), patch("compare_llm.load_rows") as load:
            with self.assertRaisesRegex(ValueError, "primary automatic EM"):
                execute_run(args)
        load.assert_not_called()


class OrchestrationTests(unittest.TestCase):
    def test_interrupted_generation_resumes_and_repairs_only_missing_pair(self):
        # Simulate all GPU work; exercise the real on-disk reservation/resume/completion path.
        from comparison_utils import append_jsonl, read_json
        rows = [example("a", 1, "g1"), example("b", 0, "g2")]
        settings = {"config": {"bootstrap_repeats": 2000, "bootstrap_seed": 42}, "classifier": {"synthetic": True}}
        plan = {"ids": ["a", "b"], "content_sha256": digest(rows)}
        args = argparse.Namespace(command="development", config="configs/llm_comparison.json", classifier_run="outputs/fake",
                                  classifier_model="models/fake", generator_lock="outputs/fake.json",
                                  run_id="20260101T000000Z_synthetic", resume=False)
        phases = []
        interrupted = [False]
        def fake_worker(phase, directory):
            phases.append(phase)
            if phase == "classifier":
                for row in rows:
                    append_jsonl(directory / "classifier_scores.jsonl", {
                        "id": row["id"], "p_answerable": .2,
                        "classifier_identity": digest(settings["classifier"])})
                return
            scores = read_jsonl(directory / "classifier_scores.jsonl")
            baseline = read_jsonl(directory / "baseline_predictions.jsonl")
            ensure_filtered(directory, rows, baseline, scores, settings, .5)
            for row in rows[len(baseline):]:
                b = prediction(row["id"], "Paris", digest(settings))
                append_jsonl(directory / "baseline_predictions.jsonl", b)
                if not interrupted[0]:
                    interrupted[0] = True
                    raise RuntimeError("Synthetic interruption between paired file writes")
                append_jsonl(directory / "filtered_predictions.jsonl", gate_replay(b, .2, .5))
        with tempfile.TemporaryDirectory() as folder, patch("compare_llm.OUTPUTS", Path(folder)), \
                patch("compare_llm.descriptor", return_value=settings), \
                patch("compare_llm.validation_plan", return_value=(rows, plan)), \
                patch("compare_llm.launch_phase", side_effect=fake_worker), \
                patch("compare_llm.show"), contextlib.redirect_stdout(io.StringIO()):
            directory = Path(folder) / args.run_id
            with self.assertRaisesRegex(RuntimeError, "Synthetic interruption"):
                execute_run(args)
            self.assertEqual(read_json(directory / "run.json")["status"], "failed")
            self.assertFalse((directory / "metrics.json").exists())
            self.assertEqual(len(read_jsonl(directory / "baseline_predictions.jsonl")), 1)
            self.assertEqual(len(read_jsonl(directory / "filtered_predictions.jsonl")), 0)
            args.resume = True
            execute_run(args)
            completed = read_json(directory / "run.json")
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["completed_unique_ids"], 2)
            self.assertEqual(completed["errors_unresolved"], 0)
            self.assertEqual(phases, ["classifier", "generator", "generator"])
            for name in ("baseline_predictions.jsonl", "filtered_predictions.jsonl", "classifier_scores.jsonl"):
                self.assertEqual([r["id"] for r in read_jsonl(directory / name)], ["a", "b"])
            self.assertTrue((directory / "execution_errors.jsonl").exists())
            with self.assertRaisesRegex(ValueError, "already completed"):
                execute_run(args)


class FakeTokenizer:
    """A character tokenizer with explicit template overhead for input-budget tests."""
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs.get("return_dict") is False, "Transformers 5.20 defaults to returning a dict"
        self.messages = messages
        return list(range(len(messages[0]["content"]) + len(messages[1]["content"]) + 10))


class InputAndCLITests(unittest.TestCase):
    def test_classifier_input_contract(self):
        from unittest.mock import Mock
        tokenizer = Mock(return_value="encoded")
        self.assertEqual(classifier_input(tokenizer, "question", "context"), "encoded")
        tokenizer.assert_called_once_with("question", "context", max_length=512, truncation="only_second",
                                          return_token_type_ids=False, return_tensors="pt", return_offsets_mapping=True)

    def test_context_only_truncation_and_no_gold_fields(self):
        tokenizer = FakeTokenizer()
        c = {"max_sequence_tokens": 100, "max_new_tokens": 10,
             "system_prompt": "Use context. NO_ANSWER if unknown.", "user_prompt": "CTX:{context} QUESTION:{question}"}
        ids, diagnostic = prompt_input(tokenizer, "UNCHANGED?", "x" * 200, c)
        self.assertLessEqual(len(ids), 90)
        self.assertTrue(diagnostic["context_truncated"])
        self.assertIn("UNCHANGED?", tokenizer.messages[1]["content"])
        self.assertNotIn("answers", str(tokenizer.messages))
        self.assertNotIn("label", str(tokenizer.messages))
        with self.assertRaisesRegex(ValueError, "Question and instructions"):
            prompt_input(tokenizer, "q" * 200, "x", c)

    def test_test_cli_has_no_subset_or_limit_and_requires_protocol(self):
        p = parser()
        test = next(action for action in p._actions if hasattr(action, "choices") and isinstance(action.choices, dict)).choices["test"]
        options = {option for action in test._actions for option in action.option_strings}
        self.assertNotIn("--limit", options)
        self.assertNotIn("--subset", options)
        self.assertNotIn("--smoke", options)
        self.assertTrue(next(a for a in test._actions if a.dest == "protocol").required)
        self.assertTrue(next(a for a in test._actions if a.dest == "local_files_only").required)
        self.assertIn("review_rubric", SCORING)


if __name__ == "__main__":
    unittest.main()
