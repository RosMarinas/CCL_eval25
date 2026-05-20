import json
import unittest

from src.harness import (
    build_formatter_input,
    build_formatter_prompt,
    decide_next_action,
    fallback_final,
    parse_reasoner_output,
    run_harness_once,
    should_skip_formatter,
    validate_reasoner_output,
)


def sample_task():
    return {
        "idx": 7,
        "title": "李端公",
        "author": "卢纶",
        "content": "故关衰草遍，离别自堪悲。",
        "qa_words": ["衰草", "衰草"],
        "qa_sents": ["故关衰草遍，离别自堪悲"],
        "choose": {
            "A": "欢快",
            "B": "闲适",
            "C": "期待",
            "D": "惜别的感伤",
        },
    }


def reasoner_output(choose_id="Ｄ"):
    return {
        "idx": 7,
        "evidence": {
            "words": {"衰草": "枯黄的草"},
            "sentences": {"故关衰草遍，离别自堪悲": "旧关衰草遍布，离别令人悲伤"},
            "emotion": ["衰草、离别、悲指向感伤"],
        },
        "draft_answer": {
            "ans_qa_words": {"衰草": "枯黄的草，烘托荒凉离别"},
            "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关长满枯草，分别本就令人悲伤"},
            "choose_id": choose_id,
        },
    }


class HarnessTest(unittest.TestCase):
    def test_parse_reasoner_output_extracts_json_object_from_wrapped_text(self):
        raw = "说明文字\n```json\n" + json.dumps(reasoner_output(), ensure_ascii=False) + "\n```"

        parsed = parse_reasoner_output(raw)

        self.assertEqual(parsed["idx"], 7)
        self.assertEqual(parsed["draft_answer"]["choose_id"], "Ｄ")


    def test_validate_reasoner_output_reports_missing_and_invalid_fields(self):
        output = reasoner_output(choose_id="Z")
        output["idx"] = 8
        output["draft_answer"]["ans_qa_words"]["衰草"] = ""
        output["draft_answer"]["ans_qa_sents"]["故关衰草遍，离别自堪悲"] = "长" * 81

        report = validate_reasoner_output(sample_task(), output)

        self.assertTrue(report["valid_json"])
        self.assertIn("衰草", report["missing_words"])
        self.assertTrue(report["invalid_choose_id"])
        self.assertIn("draft_answer.ans_qa_sents.故关衰草遍，离别自堪悲", report["overlong_fields"])
        self.assertIn("idx_mismatch", report["suspected_conflicts"])


    def test_skip_formatter_postprocesses_without_rewriting_answer_text(self):
        task = sample_task()
        output = reasoner_output()
        report = validate_reasoner_output(task, output)

        self.assertTrue(should_skip_formatter(report))
        result = run_harness_once(task, json.dumps(output, ensure_ascii=False))

        self.assertEqual(result["action"], "use_draft")
        self.assertFalse(result["formatter_called"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(
            result["final_answer"],
            {
                "idx": 7,
                "ans_qa_words": {"衰草": "枯黄的草，烘托荒凉离别"},
                "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关长满枯草，分别本就令人悲伤"},
                "choose_id": "D",
            },
        )


    def test_formatter_input_and_prompt_include_guardrails(self):
        task = sample_task()
        output = reasoner_output(choose_id="Z")
        report = validate_reasoner_output(task, output)

        formatter_input = build_formatter_input(task, output, report)
        prompt = build_formatter_prompt(formatter_input)

        self.assertEqual(formatter_input["task"], task)
        self.assertTrue(formatter_input["validator_report"]["invalid_choose_id"])
        self.assertIn("默认相信 reasoner 的 draft_answer，不要重新做题", prompt)
        self.assertIn('"validator_report"', prompt)


    def test_run_harness_calls_formatter_and_validates_final_output(self):
        task = sample_task()
        output = reasoner_output(choose_id="Z")

        def formatter_fn(formatter_input):
            self.assertTrue(formatter_input["validator_report"]["invalid_choose_id"])
            return json.dumps(
                {
                    "idx": 999,
                    "ans_qa_words": {"衰草": "枯黄的草，烘托荒凉离别"},
                    "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关长满枯草，分别本就令人悲伤"},
                    "choose_id": "D",
                    "analysis": "should be removed by validator fallback",
                },
                ensure_ascii=False,
            )

        result = run_harness_once(task, json.dumps(output, ensure_ascii=False), formatter_fn=formatter_fn)

        self.assertTrue(result["formatter_called"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["final_answer"], fallback_final(task, output))


    def test_decide_next_action_retry_and_fallback_paths(self):
        self.assertEqual(decide_next_action({"valid_json": False}), "retry_reasoner")
        self.assertEqual(decide_next_action({"valid_json": True, "missing_fields": ["draft_answer"]}), "retry_reasoner")
        self.assertEqual(
            decide_next_action({"valid_json": True, "missing_fields": [], "invalid_choose_id": True}),
            "retry_reasoner",
        )
        self.assertEqual(
            decide_next_action(
                {
                    "valid_json": True,
                    "missing_fields": [],
                    "missing_words": ["衰草"],
                    "missing_sentences": [],
                    "invalid_choose_id": False,
                    "overlong_fields": [],
                    "suspected_conflicts": [],
                }
            ),
            "call_formatter",
        )


if __name__ == "__main__":
    unittest.main()
