import json
import unittest

from src.baseline import (
    BaselineResult,
    ModelConfig,
    PromptConfig,
    build_experiment_id,
    render_answer_prompt,
    render_formatter_prompt,
    run_prompt_baseline,
)


TASK = {
    "idx": 7,
    "title": "李端公",
    "author": "卢纶",
    "content": "故关衰草遍，离别自堪悲。",
    "qa_words": ["衰草", "衰草"],
    "qa_sents": ["故关衰草遍，离别自堪悲"],
    "choose": {"A": "喜悦", "D": "惜别"},
}


class BaselineTest(unittest.TestCase):
    def test_build_experiment_id_records_group_model_backend_and_prompt(self):
        model_config = ModelConfig(
            group="P14-fast",
            model_name="Qwen/Qwen3-14B-AWQ",
            parameter_scale="14.8B",
            quantization="AWQ 4-bit",
            backend="vLLM",
            thinking_mode="non-thinking",
        )
        prompt_config = PromptConfig(prompt_type="few-shot", shot_count=3)

        self.assertEqual(
            build_experiment_id(model_config, prompt_config),
            "P14-fast-qwen3-14b-awq4-vllm-nothink-few3",
        )

    def test_render_answer_prompt_includes_few_shots_and_target_input_only(self):
        shot = {
            "input": {**TASK, "idx": 1},
            "output": {
                "idx": 1,
                "ans_qa_words": {"衰草": "枯草"},
                "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关满是枯草，离别令人悲伤"},
                "choose_id": "D",
            },
        }

        prompt = render_answer_prompt(
            TASK,
            PromptConfig(prompt_type="few-shot", shot_count=1),
            shots=[shot],
        )

        self.assertIn("示例 1 输入", prompt)
        self.assertIn("待作答输入", prompt)
        self.assertIn('"idx": 7', prompt)
        self.assertIn("现在只输出待作答输入的最终 JSON", prompt)

    def test_render_answer_prompt_requires_verbatim_input_keys(self):
        task = {
            **TASK,
            "qa_words": [" 半卷 "],
            "qa_sents": ["故关衰草遍，离别自堪悲。"],
        }

        prompt = render_answer_prompt(task, PromptConfig(prompt_type="zero-shot"))

        self.assertIn("key 必须逐字复制输入数组中的原始字符串", prompt)
        self.assertIn("包括标点、空格和全半角字符", prompt)
        self.assertIn("不能删改句末标点", prompt)
        self.assertIn('" 半卷 "', prompt)
        self.assertIn('"故关衰草遍，离别自堪悲。"', prompt)

    def test_render_formatter_prompt_uses_fmt_protocol(self):
        formatter_input = {
            "task": TASK,
            "reasoner_output": {"draft_answer": {"choose_id": "D"}},
            "validator_report": {"valid_json": True},
        }

        prompt = render_formatter_prompt(formatter_input)

        self.assertIn("formatter / verifier", prompt)
        self.assertIn("默认相信 draft_answer", prompt)
        self.assertIn('"validator_report"', prompt)

    def test_run_prompt_baseline_returns_prediction_records_with_metadata(self):
        model_config = ModelConfig(
            group="P8",
            model_name="Qwen/Qwen3-8B",
            parameter_scale="8.2B",
            quantization="bf16",
            backend="vLLM",
            thinking_mode="non-thinking",
        )
        prompt_config = PromptConfig(
            prompt_type="zero-shot",
            shot_count=0,
            decoding_params={"temperature": 0.0, "max_tokens": 256},
        )

        def generate_fn(prompt, metadata):
            self.assertIn("只输出一个合法 JSON 对象", prompt)
            self.assertEqual(
                metadata["experiment_id"], "P8-qwen3-8b-bf16-vllm-nothink-zero"
            )
            return json.dumps(
                {
                    "idx": 7,
                    "ans_qa_words": {"衰草": "枯黄的草"},
                    "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关满是枯草，离别令人悲伤"},
                    "choose_id": "D",
                },
                ensure_ascii=False,
            )

        results = run_prompt_baseline([TASK], model_config, prompt_config, generate_fn)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], BaselineResult)
        record = results[0].to_record()
        self.assertEqual(record["prediction"]["choose_id"], "D")
        self.assertIs(record["json_valid"], True)
        self.assertEqual(record["metadata"]["model_name"], "Qwen/Qwen3-8B")
        self.assertEqual(record["metadata"]["parameter_scale"], "8.2B")
        self.assertEqual(record["metadata"]["quantization"], "bf16")
        self.assertEqual(record["metadata"]["backend"], "vLLM")
        self.assertEqual(record["metadata"]["prompt_type"], "zero-shot")
        self.assertEqual(record["metadata"]["shot_count"], 0)
        self.assertEqual(record["metadata"]["decoding_params"]["max_tokens"], 256)

    def test_qwen3_think_and_nothink_have_independent_ids_and_enable_thinking_metadata(self):
        prompt_config = PromptConfig(prompt_type="zero-shot", shot_count=0)
        nothink_config = ModelConfig(
            group="P8",
            model_name="Qwen/Qwen3-8B",
            parameter_scale="8.2B",
            quantization="bf16",
            backend="vLLM",
            thinking_mode="non-thinking",
        )
        think_config = ModelConfig(
            group="P8",
            model_name="Qwen/Qwen3-8B",
            parameter_scale="8.2B",
            quantization="bf16",
            backend="vLLM",
            thinking_mode="thinking",
        )

        self.assertEqual(
            build_experiment_id(nothink_config, prompt_config),
            "P8-qwen3-8b-bf16-vllm-nothink-zero",
        )
        self.assertEqual(
            build_experiment_id(think_config, prompt_config),
            "P8-qwen3-8b-bf16-vllm-think-zero",
        )

        def generate_fn(prompt, metadata):
            enable_thinking = metadata["decoding_params"]["chat_template_kwargs"][
                "enable_thinking"
            ]
            return json.dumps(
                {
                    "idx": 7,
                    "ans_qa_words": {"衰草": "枯黄的草"},
                    "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关满是枯草，离别令人悲伤"},
                    "choose_id": "D",
                    "_enable_thinking_seen": enable_thinking,
                },
                ensure_ascii=False,
            )

        nothink_result = run_prompt_baseline(
            [TASK], nothink_config, prompt_config, generate_fn
        )[0]
        think_result = run_prompt_baseline(
            [TASK], think_config, prompt_config, generate_fn
        )[0]

        self.assertIs(
            nothink_result.metadata["decoding_params"]["chat_template_kwargs"][
                "enable_thinking"
            ],
            False,
        )
        self.assertIs(
            think_result.metadata["decoding_params"]["chat_template_kwargs"][
                "enable_thinking"
            ],
            True,
        )
        self.assertEqual(nothink_result.metadata["thinking_config"]["provider"], "qwen3")
        self.assertEqual(think_result.metadata["thinking_config"]["provider"], "qwen3")

    def test_internlm3_thinking_metadata_records_deep_thinking_placeholder(self):
        model_config = ModelConfig(
            group="P8",
            model_name="internlm/internlm3-8b-instruct",
            parameter_scale="8B",
            quantization="bf16",
            backend="vLLM",
            thinking_mode="deep thinking",
        )
        prompt_config = PromptConfig(prompt_type="zero-shot", shot_count=0)

        def generate_fn(prompt, metadata):
            self.assertEqual(
                metadata["experiment_id"],
                "P8-internlm3-8b-instruct-bf16-vllm-think-zero",
            )
            thinking_config = metadata["thinking_config"]
            self.assertEqual(thinking_config["provider"], "internlm3")
            self.assertEqual(thinking_config["mode"], "deep_thinking")
            self.assertIs(thinking_config["placeholder"], True)
            return json.dumps(
                {
                    "idx": 7,
                    "ans_qa_words": {"衰草": "枯黄的草"},
                    "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关满是枯草，离别令人悲伤"},
                    "choose_id": "D",
                },
                ensure_ascii=False,
            )

        result = run_prompt_baseline([TASK], model_config, prompt_config, generate_fn)[0]

        self.assertEqual(result.metadata["thinking_config"]["mode"], "deep_thinking")

    def test_run_prompt_baseline_fmt_renders_formatter_input_and_keeps_invalid_json(self):
        model_config = ModelConfig(
            group="FMT",
            model_name="google/gemma-4-E4B-it",
            parameter_scale="4.5B effective",
            quantization="bf16",
            backend="Transformers",
        )
        prompt_config = PromptConfig(prompt_type="fmt", shot_count=0)
        formatter_input = {
            "task": TASK,
            "reasoner_output": {"draft_answer": {"choose_id": "D"}},
            "validator_report": {"valid_json": True},
        }

        def generate_fn(prompt, metadata):
            self.assertIn("formatter / verifier", prompt)
            self.assertEqual(metadata["group"], "FMT")
            return "not json"

        result = run_prompt_baseline(
            [formatter_input],
            model_config,
            prompt_config,
            generate_fn,
        )[0]

        self.assertIsNone(result.prediction)
        self.assertIsNone(result.parsed_json)
        self.assertIs(result.json_valid, False)
        self.assertEqual(result.raw_output, "not json")


if __name__ == "__main__":
    unittest.main()
