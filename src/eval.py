"""Evaluation records and error classification for CCL25 experiments."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any

try:
    from src.schema import validate_output
except ImportError:  # I1 owns schema.py; keep eval importable while it is absent.
    validate_output = None


RESULT_FIELDS = [
    "experiment_id",
    "group",
    "model_role",
    "reasoner_model",
    "formatter_model",
    "param_total_b",
    "quantization",
    "backend",
    "mode",
    "prompt_type",
    "shot_count",
    "decode_params",
    "dev_split_id",
    "sample_count",
    "word_score",
    "translation_score",
    "emotion_score",
    "total_score",
    "json_error_rate",
    "hard_json_error_rate",
    "format_style_error_rate",
    "avg_latency_ms",
    "p95_latency_ms",
    "formatter_call_rate",
    "formatter_regression_rate",
    "retry_rate",
    "fallback_rate",
    "notes",
]

DETAIL_FIELDS = [
    "experiment_id",
    "idx",
    "raw_output",
    "parsed_json",
    "json_valid",
    "json_error_categories",
    "word_score",
    "translation_score",
    "emotion_score",
    "total_score",
    "latency_ms",
    "reasoner_latency_ms",
    "formatter_latency_ms",
    "formatter_called",
    "reasoner_retried",
    "fallback_used",
    "draft_answer",
    "final_answer",
]

JSON_ERROR_CATEGORIES = [
    "parse_error",
    "extra_text",
    "thinking_trace_leak",
    "missing_top_field",
    "extra_top_field",
    "idx_mismatch",
    "wrong_field_type",
    "missing_word_key",
    "extra_word_key",
    "missing_sentence_key",
    "extra_sentence_key",
    "empty_required_answer",
    "invalid_choose_id",
    "overlong_word_answer",
    "overlong_sentence_answer",
    "non_chinese_or_unusable",
]

HARD_JSON_ERRORS = {
    "parse_error",
    "missing_top_field",
    "idx_mismatch",
    "wrong_field_type",
    "invalid_choose_id",
}

# json_error_rate 的核心构成 = HARD_JSON_ERRORS + coverage + non_chinese
CORE_JSON_ERRORS = HARD_JSON_ERRORS | {
    "missing_word_key",
    "missing_sentence_key",
    "empty_required_answer",
    "non_chinese_or_unusable",
}

# 格式/风格问题，剥离后可用的错误
FORMAT_STYLE_ERRORS = {
    "extra_text",
    "thinking_trace_leak",
    "extra_top_field",
    "extra_word_key",
    "extra_sentence_key",
    "overlong_word_answer",
    "overlong_sentence_answer",
}

_SCHEMA_TO_EVAL_ERROR = {
    "missing_output_fields": "missing_top_field",
    "unexpected_output_fields": "extra_top_field",
    "missing_ans_qa_words": "missing_word_key",
    "unexpected_ans_qa_words": "extra_word_key",
    "missing_ans_qa_sents": "missing_sentence_key",
    "unexpected_ans_qa_sents": "extra_sentence_key",
    "empty_ans_qa_words": "empty_required_answer",
    "empty_ans_qa_sents": "empty_required_answer",
    "invalid_ans_qa_words": "wrong_field_type",
    "invalid_ans_qa_sents": "wrong_field_type",
    "invalid_choose_id_type": "wrong_field_type",
}

TOP_FIELDS = {"idx", "ans_qa_words", "ans_qa_sents", "choose_id"}
THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
EXPLICIT_THINKING_RE = re.compile(
    r"(^|\n)\s*(thinking|reasoning|analysis|thought|思考|推理|分析)\s*[:：]",
    re.IGNORECASE,
)
FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse a single JSON object, preserving an extra_text label for wrappers."""

    text = "" if raw is None else str(raw)
    stripped = text.strip()
    if not stripped:
        return None, ["parse_error"]

    errors: list[str] = []
    if _has_thinking_trace(stripped):
        errors.extend(["extra_text", "thinking_trace_leak"])

    think_stripped = THINK_TAG_RE.sub("", stripped).strip()
    candidates = [think_stripped]
    fenced = _extract_fenced_json(think_stripped)
    if fenced is not None:
        candidates.append(fenced)
    bracketed = _extract_first_json_object(think_stripped)
    if bracketed is not None:
        candidates.append(bracketed)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            return None, ["parse_error"]
        if candidate != stripped:
            errors.append("extra_text")
        return parsed, _ordered_errors(errors)

    return None, _ordered_errors(errors + ["parse_error"])


def classify_json_errors(
    raw_output: str,
    task: dict[str, Any],
    parsed_json: dict[str, Any] | None = None,
    parse_errors: list[str] | None = None,
) -> list[str]:
    """Return ordered JSON/schema error labels for one model output.

    If *parse_errors* and *parsed_json* are both provided, the raw-output
    extraction step is skipped and the pre-computed labels are reused.
    """

    if parse_errors is not None and parsed_json is not None:
        errors = list(parse_errors)
    else:
        parsed_from_raw, errors = parse_json_object(raw_output)
        if parsed_json is None:
            parsed_json = parsed_from_raw
    if parsed_json is None:
        return _ordered_errors(errors)

    errors.extend(_local_schema_errors(parsed_json, task))
    errors.extend(_validate_output_errors(parsed_json, task))
    return _ordered_errors(errors)


def make_experiment_record(**kwargs: Any) -> dict[str, Any]:
    """Build one row of the main experiment result table."""

    defaults = {
        "reasoner_model": "",
        "formatter_model": "",
        "formatter_call_rate": None,
        "formatter_regression_rate": None,
        "retry_rate": 0,
        "fallback_rate": 0,
        "notes": "",
    }
    values = {**defaults, **kwargs}
    return OrderedDict((field, values.get(field)) for field in RESULT_FIELDS)


def make_sample_record(**kwargs: Any) -> dict[str, Any]:
    """Build one row of the per-sample detail table."""

    defaults = {
        "raw_output": "",
        "parsed_json": None,
        "json_valid": False,
        "json_error_categories": [],
        "word_score": None,
        "translation_score": None,
        "emotion_score": None,
        "total_score": None,
        "latency_ms": None,
        "reasoner_latency_ms": None,
        "formatter_latency_ms": None,
        "formatter_called": False,
        "reasoner_retried": False,
        "fallback_used": False,
        "draft_answer": None,
        "final_answer": None,
    }
    values = {**defaults, **kwargs}
    return OrderedDict((field, values.get(field)) for field in DETAIL_FIELDS)


def compute_formatter_regression(
    draft_scores: list[dict[str, float]],
    final_scores: list[dict[str, float]],
    draft_errors: list[list[str]] | None = None,
    final_errors: list[list[str]] | None = None,
    threshold: float = 0.05,
) -> dict[str, float | int]:
    """Summarize formatter regressions from paired draft/final scores."""

    if len(draft_scores) != len(final_scores):
        raise ValueError("draft_scores and final_scores must have the same length")
    count = len(draft_scores)
    draft_errors = draft_errors or [[] for _ in range(count)]
    final_errors = final_errors or [[] for _ in range(count)]
    if len(draft_errors) != count or len(final_errors) != count:
        raise ValueError("draft_errors and final_errors must match score length")
    if count == 0:
        return {
            "sample_count": 0,
            "formatter_regression_rate": 0,
            "formatter_word_regression_rate": 0,
            "formatter_translation_regression_rate": 0,
            "formatter_emotion_regression_rate": 0,
            "formatter_json_regression_rate": 0,
            "formatter_fix_rate": 0,
            "formatter_net_gain": 0,
        }

    word_regressions = 0
    translation_regressions = 0
    emotion_regressions = 0
    json_regressions = 0
    fixes = 0
    net_gain = 0.0
    any_regressions = 0

    for draft, final, draft_err, final_err in zip(
        draft_scores, final_scores, draft_errors, final_errors
    ):
        word = _task_regressed(draft, final, "word_score", threshold)
        translation = _task_regressed(draft, final, "translation_score", threshold)
        emotion = _task_regressed(draft, final, "emotion_score", threshold)
        total = _score(draft, "total_score") - _score(final, "total_score") >= threshold
        json_regression = not _has_hard_json_error(draft_err) and _has_hard_json_error(final_err)

        word_regressions += int(word)
        translation_regressions += int(translation)
        emotion_regressions += int(emotion)
        json_regressions += int(json_regression)
        any_regressions += int(word or translation or emotion or total or json_regression)

        draft_total = _score(draft, "total_score")
        final_total = _score(final, "total_score")
        net_gain += final_total - draft_total
        if _has_json_or_coverage_error(draft_err) and not _has_json_or_coverage_error(final_err):
            fixes += int(final_total >= draft_total)

    return {
        "sample_count": count,
        "formatter_regression_rate": any_regressions / count,
        "formatter_word_regression_rate": word_regressions / count,
        "formatter_translation_regression_rate": translation_regressions / count,
        "formatter_emotion_regression_rate": emotion_regressions / count,
        "formatter_json_regression_rate": json_regressions / count,
        "formatter_fix_rate": fixes / count,
        "formatter_net_gain": net_gain / count,
    }


def compute_json_error_rates(error_categories: list[list[str]]) -> dict[str, float | int]:
    """Summarize JSON error rates: core (output unusable) and format (output usable but messy)."""

    count = len(error_categories)
    if count == 0:
        return {
            "sample_count": 0,
            "json_error_rate": 0,
            "hard_json_error_rate": 0,
            "format_style_error_rate": 0,
        }

    json_errors = 0       # 核心错误（输出不可用）
    hard_errors = 0       # 严重错误子集
    format_errors = 0     # 格式问题（输出仍可用）

    for categories in error_categories:
        category_set = set(categories)
        if category_set & CORE_JSON_ERRORS:
            json_errors += 1
        if category_set & HARD_JSON_ERRORS:
            hard_errors += 1
        if category_set & FORMAT_STYLE_ERRORS:
            format_errors += 1

    return {
        "sample_count": count,
        "json_error_rate": json_errors / count,
        "hard_json_error_rate": hard_errors / count,
        "format_style_error_rate": format_errors / count,
    }


def task_error_template(task_type: str) -> dict[str, Any]:
    """Return the documented per-task manual error analysis template."""

    templates = {
        "word": OrderedDict(
            [
                ("idx", None),
                ("experiment_id", ""),
                ("target_word", ""),
                ("gold_or_reference", ""),
                ("prediction", ""),
                (
                    "error_type",
                    [
                        "missing",
                        "literal_only",
                        "context_misread",
                        "over_explained",
                        "wrong_sense",
                        "format_only",
                    ],
                ),
                ("evidence_issue", ""),
                ("fix_hint", ""),
            ]
        ),
        "translation": OrderedDict(
            [
                ("idx", None),
                ("target_sentence", ""),
                ("gold_or_reference", ""),
                ("prediction", ""),
                (
                    "error_type",
                    [
                        "missing",
                        "partial_translation",
                        "syntax_misread",
                        "imagery_loss",
                        "added_appreciation",
                        "too_long",
                        "wrong_subject",
                    ],
                ),
                ("affected_score", None),
                ("fix_hint", ""),
            ]
        ),
        "emotion": OrderedDict(
            [
                ("idx", None),
                ("choose", {}),
                ("gold_choose_id", ""),
                ("pred_choose_id", ""),
                (
                    "error_type",
                    [
                        "opposite_emotion",
                        "near_option_confusion",
                        "local_cue_overfit",
                        "ignored_title_author",
                        "formatter_changed",
                        "invalid_option",
                    ],
                ),
                ("key_evidence", ""),
                ("model_evidence", ""),
                ("fix_hint", ""),
            ]
        ),
        "json_harness": OrderedDict(
            [
                ("idx", None),
                ("json_error_categories", []),
                (
                    "stage",
                    [
                        "reasoner",
                        "validator",
                        "formatter",
                        "final_validator",
                        "fallback",
                    ],
                ),
                ("raw_output_excerpt", ""),
                ("validator_report", ""),
                ("formatter_called", False),
                ("fallback_used", False),
                ("fix_hint", ""),
            ]
        ),
    }
    if task_type not in templates:
        raise ValueError("task_type must be one of: word, translation, emotion, json_harness")
    return dict(templates[task_type])


def _local_schema_errors(parsed: dict[str, Any], task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = TOP_FIELDS - parsed.keys()
    if missing:
        errors.append("missing_top_field")
    if parsed.keys() - TOP_FIELDS:
        errors.append("extra_top_field")

    if "idx" in parsed and parsed.get("idx") != task.get("idx"):
        errors.append("idx_mismatch")

    word_answers = parsed.get("ans_qa_words")
    sent_answers = parsed.get("ans_qa_sents")
    choose_id = parsed.get("choose_id")
    if "ans_qa_words" in parsed and not isinstance(word_answers, dict):
        errors.append("wrong_field_type")
    if "ans_qa_sents" in parsed and not isinstance(sent_answers, dict):
        errors.append("wrong_field_type")
    if "choose_id" in parsed and not isinstance(choose_id, str):
        errors.append("wrong_field_type")

    if isinstance(word_answers, dict):
        errors.extend(_answer_key_errors(word_answers, _unique(task.get("qa_words", [])), "word"))
        errors.extend(_answer_value_errors(word_answers, "word"))
    if isinstance(sent_answers, dict):
        errors.extend(_answer_key_errors(sent_answers, _unique(task.get("qa_sents", [])), "sentence"))
        errors.extend(_answer_value_errors(sent_answers, "sentence"))

    choices = task.get("choose") or {}
    if isinstance(choose_id, str):
        if choices and choose_id not in {str(key) for key in choices.keys()}:
            errors.append("invalid_choose_id")
        if not choices and choose_id != "":
            errors.append("invalid_choose_id")

    return errors


def _answer_key_errors(answer_map: dict[Any, Any], targets: list[str], kind: str) -> list[str]:
    errors: list[str] = []
    keys = {str(key) for key in answer_map.keys()}
    target_keys = {str(key) for key in targets}
    if target_keys - keys:
        errors.append(f"missing_{kind}_key")
    if keys - target_keys:
        errors.append(f"extra_{kind}_key")
    return errors


def _answer_value_errors(answer_map: dict[Any, Any], kind: str) -> list[str]:
    errors: list[str] = []
    overlong_label = f"overlong_{kind}_answer"
    limit = 40 if kind == "word" else 80
    for value in answer_map.values():
        if value is None or value == {} or value == "":
            errors.append("empty_required_answer")
            continue
        if not isinstance(value, str):
            errors.append("wrong_field_type")
            continue
        if _cjk_len(value) > limit:
            errors.append(overlong_label)
        if not _contains_cjk(value):
            errors.append("non_chinese_or_unusable")
    return errors


def _validate_output_errors(parsed: dict[str, Any], task: dict[str, Any]) -> list[str]:
    if validate_output is None:
        return []
    try:
        result = validate_output(parsed, task)
    except TypeError:
        try:
            result = validate_output(parsed)
        except Exception:
            return []
    except Exception:
        return []
    if result is True or result is None:
        return []
    errors: list[str] = []
    raw_errors = []
    if isinstance(result, list):
        raw_errors = result
    elif isinstance(result, dict):
        raw_errors = result.get("errors") or result.get("error_categories") or []
    for err in raw_errors:
        mapped = _SCHEMA_TO_EVAL_ERROR.get(err)
        if mapped and mapped not in errors:
            errors.append(mapped)
    return errors


def _ordered_errors(errors: list[str]) -> list[str]:
    seen = set()
    result = []
    for category in JSON_ERROR_CATEGORIES:
        if category in errors and category not in seen:
            result.append(category)
            seen.add(category)
    return result


def _has_thinking_trace(text: str) -> bool:
    return bool(THINK_TAG_RE.search(text) or EXPLICIT_THINKING_RE.search(text))


def _extract_fenced_json(text: str) -> str | None:
    match = FENCE_RE.match(text)
    if not match:
        return None
    return match.group(1).strip()


def _extract_first_json_object(text: str) -> str | None:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return text[start : start + end]
    return None


def _unique(values: Any) -> list[str]:
    result = []
    seen = set()
    for value in values or []:
        text = str(value)
        if text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _cjk_len(value: str) -> int:
    return sum(1 for char in value if "\u4e00" <= char <= "\u9fff")


def _score(scores: dict[str, float], key: str) -> float:
    value = scores.get(key, 0)
    return float(value or 0)


def _task_regressed(
    draft: dict[str, float],
    final: dict[str, float],
    key: str,
    threshold: float,
) -> bool:
    return _score(draft, key) >= 1.0 and _score(final, key) <= 1.0 - threshold


def _has_hard_json_error(errors: list[str]) -> bool:
    return bool(set(errors) & HARD_JSON_ERRORS)


def _has_json_or_coverage_error(errors: list[str]) -> bool:
    return bool(set(errors) & CORE_JSON_ERRORS)
