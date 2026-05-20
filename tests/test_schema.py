import unittest

from src.schema import (
    build_output_from_training,
    normalize_input,
    validate_input,
    validate_output,
)


class SchemaTest(unittest.TestCase):
    def test_normalize_input_maps_aliases_and_choose_list(self):
        raw = {
            "index": "sample-002",
            "title": " 登鹳雀楼 ",
            "poem": " 白日依山尽，黄河入海流。\n欲穷千里目，更上一层楼。 ",
            "target_words": [" 千里目 "],
            "target_sents": [" 欲穷千里目，更上一层楼。 "],
            "options": ["开阔昂扬", "哀怨低沉"],
        }

        normalized = normalize_input(raw)

        self.assertEqual(
            normalized,
            {
                "idx": "sample-002",
                "title": "登鹳雀楼",
                "author": "",
                "content": "白日依山尽，黄河入海流。\n欲穷千里目，更上一层楼。",
                "qa_words": ["千里目"],
                "qa_sents": ["欲穷千里目，更上一层楼。"],
                "choose": {"A": "开阔昂扬", "B": "哀怨低沉"},
            },
        )

    def test_validate_input_reports_missing_content_and_empty_sections(self):
        result = validate_input({"idx": 1, "content": " ", "choose": {"X": ""}})

        self.assertFalse(result["valid"])
        self.assertIn("invalid_input_missing_content", result["errors"])
        self.assertIn("empty_qa_words", result["warnings"])
        self.assertIn("empty_qa_sents", result["warnings"])
        self.assertIn("empty_choose_text", result["warnings"])
        self.assertIn("non_standard_choose_keys", result["warnings"])

    def test_validate_output_rejects_extra_keys_and_missing_coverage(self):
        task = normalize_input(
            {
                "idx": 3,
                "content": "春风又绿江南岸。",
                "qa_words": ["春风", "春风"],
                "qa_sents": ["春风又绿江南岸。"],
                "choose": {"A": "喜悦"},
            }
        )
        output = {
            "idx": 3,
            "ans_qa_words": {},
            "ans_qa_sents": {"春风又绿江南岸。": "春风又吹绿了江南岸"},
            "choose_id": "B",
            "content": "should not be here",
        }

        result = validate_output(output, task)

        self.assertFalse(result["valid"])
        self.assertIn("unexpected_output_fields", result["errors"])
        self.assertIn("missing_ans_qa_words", result["errors"])
        self.assertIn("invalid_choose_id", result["errors"])

    def test_build_output_from_training_uses_keywords_only_and_flags_unmapped_fields(self):
        raw = {
            "idx": 1,
            "keywords": {"故关": "旧日关塞", "风尘": "战乱漂泊", "多余": "忽略"},
            "trans": "全诗译文",
            "emotion": "惜别",
        }
        task = {
            "idx": 1,
            "qa_words": ["故关", "风尘"],
            "qa_sents": ["故关衰草遍，离别自堪悲"],
            "choose": {"A": "欢快", "D": "惜别"},
        }

        output, issues = build_output_from_training(raw, task)

        self.assertEqual(
            output,
            {
                "idx": 1,
                "ans_qa_words": {"故关": "旧日关塞", "风尘": "战乱漂泊"},
                "ans_qa_sents": {},
                "choose_id": "",
            },
        )
        self.assertIn("unmapped_trans", issues)
        self.assertIn("unmapped_emotion", issues)


if __name__ == "__main__":
    unittest.main()
