import json
import unittest

from src.eval import (
    DETAIL_FIELDS,
    JSON_ERROR_CATEGORIES,
    RESULT_FIELDS,
    classify_json_errors,
    compute_formatter_regression,
    compute_json_error_rates,
    make_experiment_record,
    make_sample_record,
    parse_json_object,
    task_error_template,
)


SAMPLE_TASK = {
    "idx": 7,
    "qa_words": ["春风", "春风"],
    "qa_sents": ["春风又绿江南岸。"],
    "choose": {"A": "喜悦", "B": "悲伤"},
}


class EvalRecorderTest(unittest.TestCase):
    def test_parse_json_object_accepts_object_and_flags_wrapped_text(self):
        parsed, errors = parse_json_object('{"idx": 7, "choose_id": "A"}')
        self.assertEqual(parsed, {"idx": 7, "choose_id": "A"})
        self.assertEqual(errors, [])

        parsed, errors = parse_json_object('```json\n{"idx": 7}\n```')
        self.assertEqual(parsed, {"idx": 7})
        self.assertEqual(errors, ["extra_text"])

        parsed, errors = parse_json_object("not json")
        self.assertIsNone(parsed)
        self.assertEqual(errors, ["parse_error"])

    def test_classify_json_errors_reports_schema_and_task_mismatches(self):
        raw = json.dumps(
            {
                "idx": 8,
                "ans_qa_words": {"春风": "", "多余": "答案"},
                "ans_qa_sents": {},
                "choose_id": "C",
                "analysis": "not allowed",
            },
            ensure_ascii=False,
        )

        errors = classify_json_errors(raw, SAMPLE_TASK)

        self.assertIn("idx_mismatch", errors)
        self.assertIn("extra_top_field", errors)
        self.assertIn("extra_word_key", errors)
        self.assertIn("missing_sentence_key", errors)
        self.assertIn("empty_required_answer", errors)
        self.assertIn("invalid_choose_id", errors)
        self.assertTrue(set(errors).issubset(JSON_ERROR_CATEGORIES))

    def test_record_builders_emit_documented_fields(self):
        experiment = make_experiment_record(
            experiment_id="H2-test",
            group="H2",
            model_role="harness",
            reasoner_model="reasoner",
            formatter_model="formatter",
            param_total_b=12.0,
            quantization="bf16",
            backend="vllm",
            mode="nothink",
            prompt_type="few",
            shot_count=3,
            decode_params={"temperature": 0},
            dev_split_id="dev_main",
            sample_count=2,
            word_score=0.8,
            translation_score=0.7,
            emotion_score=1.0,
            total_score=0.82,
            json_error_rate=0.1,
            avg_latency_ms=100,
            p95_latency_ms=150,
        )
        self.assertEqual(list(experiment.keys()), RESULT_FIELDS)
        self.assertIsNone(experiment["formatter_call_rate"])
        self.assertEqual(experiment["retry_rate"], 0)

        sample = make_sample_record(
            experiment_id="H2-test",
            idx=7,
            raw_output="{}",
            parsed_json={},
            json_valid=False,
            json_error_categories=["missing_top_field"],
        )
        self.assertEqual(list(sample.keys()), DETAIL_FIELDS)
        self.assertIsNone(sample["latency_ms"])
        self.assertFalse(sample["formatter_called"])

    def test_compute_formatter_regression_counts_score_json_and_fix_rates(self):
        summary = compute_formatter_regression(
            draft_scores=[
                {"word_score": 1.0, "translation_score": 1.0, "emotion_score": 1.0, "total_score": 1.0},
                {"word_score": 0.0, "translation_score": 0.5, "emotion_score": 1.0, "total_score": 0.5},
                {"word_score": 0.5, "translation_score": 0.5, "emotion_score": 0.5, "total_score": 0.5},
            ],
            final_scores=[
                {"word_score": 0.0, "translation_score": 1.0, "emotion_score": 1.0, "total_score": 0.6},
                {"word_score": 0.1, "translation_score": 0.5, "emotion_score": 1.0, "total_score": 0.55},
                {"word_score": 0.5, "translation_score": 0.5, "emotion_score": 0.5, "total_score": 0.5},
            ],
            draft_errors=[[], ["parse_error"], []],
            final_errors=[[], [], ["idx_mismatch"]],
            threshold=0.05,
        )

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["formatter_regression_rate"], 2 / 3)
        self.assertEqual(summary["formatter_word_regression_rate"], 1 / 3)
        self.assertEqual(summary["formatter_json_regression_rate"], 1 / 3)
        self.assertEqual(summary["formatter_fix_rate"], 1 / 3)
        self.assertEqual(summary["formatter_net_gain"], -0.11666666666666665)

    def test_compute_json_error_rates_reports_grouped_rates(self):
        summary = compute_json_error_rates(
            [
                [],
                ["parse_error", "missing_word_key"],
                ["extra_text", "overlong_word_answer"],
                ["empty_required_answer"],
            ]
        )

        self.assertEqual(summary["sample_count"], 4)
        self.assertEqual(summary["json_error_rate"], 3 / 4)
        self.assertEqual(summary["hard_json_error_rate"], 1 / 4)
        self.assertEqual(summary["coverage_error_rate"], 2 / 4)
        self.assertEqual(summary["format_style_error_rate"], 1 / 4)

    def test_task_error_template_returns_expected_shapes(self):
        word_template = task_error_template("word")
        harness_template = task_error_template("json_harness")

        self.assertEqual(
            word_template["error_type"],
            [
                "missing",
                "literal_only",
                "context_misread",
                "over_explained",
                "wrong_sense",
                "format_only",
            ],
        )
        self.assertEqual(
            harness_template["stage"],
            [
                "reasoner",
                "validator",
                "formatter",
                "final_validator",
                "fallback",
            ],
        )


if __name__ == "__main__":
    unittest.main()
