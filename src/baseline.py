from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


GenerateFn = Callable[[str, dict[str, Any]], str]

BASELINE_GROUPS = {"P14", "P14-fast", "P8", "FMT"}
ANSWER_PROMPT_TYPES = {"zero-shot", "few-shot"}
FMT_PROMPT_TYPES = {"fmt", "formatter"}


@dataclass(frozen=True)
class ModelConfig:
    group: str
    model_name: str
    parameter_scale: str
    quantization: str
    backend: str
    thinking_mode: str = "non-thinking"


@dataclass(frozen=True)
class PromptConfig:
    prompt_type: str
    shot_count: int = 0
    decoding_params: dict[str, Any] = field(default_factory=dict)
    prompt_name: str = "default"


@dataclass(frozen=True)
class BaselineResult:
    idx: Any
    experiment_id: str
    raw_output: str
    parsed_json: dict[str, Any] | None
    prediction: dict[str, Any] | None
    json_valid: bool
    latency_ms: float
    metadata: dict[str, Any]
    validation_error: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "idx": self.idx,
            "experiment_id": self.experiment_id,
            "prediction": self.prediction,
            "raw_output": self.raw_output,
            "parsed_json": self.parsed_json,
            "json_valid": self.json_valid,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }
        if self.validation_error:
            record["validation_error"] = self.validation_error
        return record


def build_experiment_id(model_config: ModelConfig, prompt_config: PromptConfig) -> str:
    _validate_configs(model_config, prompt_config)
    model_slug = _model_slug(model_config.model_name)
    quant_slug = _quantization_slug(model_config.quantization)
    backend_slug = _slug(model_config.backend)
    thinking_slug = _thinking_slug(model_config.thinking_mode)
    prompt_slug = _prompt_slug(prompt_config)
    return "-".join(
        [
            model_config.group,
            model_slug,
            quant_slug,
            backend_slug,
            thinking_slug,
            prompt_slug,
        ]
    )


def render_answer_prompt(
    task: dict[str, Any],
    prompt_config: PromptConfig,
    shots: list[dict[str, Any]] | None = None,
) -> str:
    if prompt_config.prompt_type not in ANSWER_PROMPT_TYPES:
        raise ValueError(f"Unsupported answer prompt type: {prompt_config.prompt_type}")

    task_json = _json_dumps(task)
    instructions = _answer_instructions()
    if prompt_config.prompt_type == "zero-shot":
        return (
            f"{instructions}\n\n"
            f"输入：\n{task_json}\n\n"
            "现在只输出最终 JSON："
        )

    shot_blocks = []
    selected_shots = list(shots or [])[: prompt_config.shot_count]
    for index, shot in enumerate(selected_shots, start=1):
        shot_blocks.append(
            f"示例 {index} 输入：\n{_json_dumps(shot['input'])}\n\n"
            f"示例 {index} 输出：\n{_json_dumps(shot['output'])}"
        )
    examples = "\n\n".join(shot_blocks)
    if examples:
        examples = f"\n\n{examples}"
    return (
        f"{instructions}{examples}\n\n"
        f"待作答输入：\n{task_json}\n\n"
        "现在只输出待作答输入的最终 JSON："
    )


def render_formatter_prompt(formatter_input: dict[str, Any]) -> str:
    return (
        "你是古诗词理解任务的 formatter / verifier。\n\n"
        "输入包括 task、reasoner_output 和 validator_report。\n"
        "你的任务是把 reasoner_output.draft_answer 整理成最终提交 JSON，并做轻量校验。\n\n"
        "必须遵守：\n"
        "1. 默认相信 draft_answer，不要重新做题。\n"
        "2. 不要输出推理过程、解释、Markdown 或代码块，只输出一个 JSON 对象。\n"
        "3. 最终 JSON 只能包含 idx、ans_qa_words、ans_qa_sents、choose_id 四个字段。\n"
        "4. idx 必须使用 task.idx。\n"
        "5. ans_qa_words 的 key 必须覆盖 task.qa_words 去重后的词语。\n"
        "6. ans_qa_sents 的 key 必须覆盖 task.qa_sents 去重后的句子。\n"
        "7. choose_id 必须来自 task.choose 的选项 ID。\n"
        "8. 只在 validator_report 指出结构问题、缺项、非法选项、过长答案或明显证据冲突时，才轻微修改 draft_answer。\n"
        "9. 如果缺项无法根据 draft_answer 和 evidence 补齐，使用空字符串占位，不要发明长解释。\n\n"
        f"输入：\n{_json_dumps(formatter_input)}\n\n"
        "现在只输出最终 JSON："
    )


def run_prompt_baseline(
    tasks: Iterable[dict[str, Any]],
    model_config: ModelConfig,
    prompt_config: PromptConfig,
    generate_fn: GenerateFn,
    shots: list[dict[str, Any]] | None = None,
) -> list[BaselineResult]:
    _validate_configs(model_config, prompt_config)
    experiment_id = build_experiment_id(model_config, prompt_config)
    results = []
    for item in tasks:
        task = _task_for_item(item, model_config.group)
        normalized_task = _normalize_task(task)
        prompt = (
            render_formatter_prompt(item)
            if model_config.group == "FMT" or prompt_config.prompt_type in FMT_PROMPT_TYPES
            else render_answer_prompt(normalized_task, prompt_config, shots=shots)
        )
        metadata = _metadata(
            model_config=model_config,
            prompt_config=prompt_config,
            experiment_id=experiment_id,
            idx=normalized_task.get("idx"),
        )

        start = time.perf_counter()
        raw_output = generate_fn(prompt, metadata)
        latency_ms = (time.perf_counter() - start) * 1000

        parsed_json, parse_error = _parse_json_object(raw_output)
        validation_error = parse_error
        json_valid = parsed_json is not None
        if parsed_json is not None:
            schema_valid, schema_error = _validate_output_if_available(
                normalized_task,
                parsed_json,
            )
            json_valid = schema_valid
            validation_error = schema_error

        results.append(
            BaselineResult(
                idx=normalized_task.get("idx"),
                experiment_id=experiment_id,
                raw_output=raw_output,
                parsed_json=parsed_json,
                prediction=parsed_json if json_valid else None,
                json_valid=json_valid,
                latency_ms=latency_ms,
                metadata=metadata,
                validation_error=validation_error,
            )
        )
    return results


def _answer_instructions() -> str:
    return (
        "你需要完成古诗词理解任务。请根据输入诗歌、目标词语、目标句子和情感选项，直接生成最终答案 JSON。\n\n"
        "输出要求：\n"
        "- 只输出一个合法 JSON 对象。\n"
        "- 不要输出 Markdown 代码块。\n"
        "- 不要输出解释、分析、证据、草稿或任何 JSON 之外的文字。\n"
        "- JSON 字段必须且只能包含：idx、ans_qa_words、ans_qa_sents、choose_id。\n"
        "- idx 必须与输入 idx 完全一致。\n"
        "- ans_qa_words 是对象，key 必须使用 qa_words 中的原词；重复词语只输出一个 key；value 是该词在诗中的简洁解释。\n"
        "- ans_qa_sents 是对象，key 必须使用 qa_sents 中的原句；重复句子只输出一个 key；value 是该句的简洁现代汉语翻译。\n"
        "- choose_id 必须从 choose 的选项 ID 中选择一个最符合全诗情感的选项。"
    )


def _metadata(
    model_config: ModelConfig,
    prompt_config: PromptConfig,
    experiment_id: str,
    idx: Any,
) -> dict[str, Any]:
    return {
        "idx": idx,
        "experiment_id": experiment_id,
        "group": model_config.group,
        "model_name": model_config.model_name,
        "parameter_scale": model_config.parameter_scale,
        "quantization": model_config.quantization,
        "backend": model_config.backend,
        "thinking_mode": model_config.thinking_mode,
        "prompt_type": prompt_config.prompt_type,
        "prompt_name": prompt_config.prompt_name,
        "shot_count": prompt_config.shot_count,
        "decoding_params": dict(prompt_config.decoding_params),
    }


def _validate_configs(model_config: ModelConfig, prompt_config: PromptConfig) -> None:
    if model_config.group not in BASELINE_GROUPS:
        raise ValueError(f"Unsupported baseline group: {model_config.group}")
    if prompt_config.shot_count < 0:
        raise ValueError("shot_count must be non-negative")
    if model_config.group == "FMT":
        if prompt_config.prompt_type not in FMT_PROMPT_TYPES:
            raise ValueError("FMT group requires fmt prompt_type")
    elif prompt_config.prompt_type not in ANSWER_PROMPT_TYPES:
        raise ValueError(f"Unsupported prompt type: {prompt_config.prompt_type}")


def _task_for_item(item: dict[str, Any], group: str) -> dict[str, Any]:
    if group == "FMT":
        task = item.get("task")
        if not isinstance(task, dict):
            raise ValueError("FMT item must contain a task object")
        return task
    return item


def _normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    try:
        from src.schema import normalize_input
    except ImportError:
        return task
    return normalize_input(task)


def _validate_output_if_available(
    task: dict[str, Any],
    output: dict[str, Any],
) -> tuple[bool, str | None]:
    try:
        from src.schema import validate_output
    except ImportError:
        return True, None

    for args in ((output, task), (task, output), (output,)):
        try:
            result = validate_output(*args)
            return _schema_result_to_bool(result)
        except TypeError:
            continue
        except Exception as exc:  # noqa: BLE001 - schema validator owns exact errors.
            return False, str(exc)
    return False, "validate_output signature is unsupported"


def _schema_result_to_bool(result: Any) -> tuple[bool, str | None]:
    if result is None:
        return True, None
    if isinstance(result, bool):
        return result, None if result else "schema validation failed"
    if isinstance(result, tuple) and result:
        valid = bool(result[0])
        return valid, None if valid else str(result[1:])
    if isinstance(result, dict):
        if "valid" in result or "ok" in result:
            valid = bool(result.get("valid", result.get("ok")))
        else:
            invalid_flags = [
                key for key, value in result.items() if key.startswith("invalid_") and value
            ]
            missing_items = [
                value
                for key, value in result.items()
                if key.startswith("missing_") and value
            ]
            valid_json = result.get("valid_json", True)
            valid = bool(valid_json) and not invalid_flags and not missing_items
        return valid, None if valid else _json_dumps(result)
    return bool(result), None if result else "schema validation failed"


def _parse_json_object(raw_output: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_output.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return None, "raw output is not a JSON object"
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return None, str(exc)
    if not isinstance(parsed, dict):
        return None, "raw output JSON is not an object"
    return parsed, None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def _prompt_slug(prompt_config: PromptConfig) -> str:
    if prompt_config.prompt_type == "zero-shot":
        return "zero"
    if prompt_config.prompt_type == "few-shot":
        return f"few{prompt_config.shot_count}"
    return "jsonfix"


def _model_slug(model_name: str) -> str:
    slug = _slug(model_name.rsplit("/", maxsplit=1)[-1])
    return re.sub(r"-(awq|gptq|int4|int8)$", "", slug)


def _quantization_slug(quantization: str) -> str:
    text = quantization.lower()
    if "awq" in text:
        return "awq4" if "4" in text or "bit" in text else "awq"
    if "bf16" in text or "bfloat16" in text:
        return "bf16"
    if "fp16" in text:
        return "fp16"
    if "fp8" in text:
        return "fp8"
    return _slug(quantization)


def _thinking_slug(thinking_mode: str) -> str:
    text = thinking_mode.lower()
    if "non" in text or "no-think" in text or "nothink" in text:
        return "nothink"
    if "normal" in text:
        return "normal"
    if "deep" in text or "think" in text:
        return "think"
    return _slug(thinking_mode)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"
