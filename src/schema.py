from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


INPUT_FIELDS = {"idx", "title", "author", "content", "qa_words", "qa_sents", "choose"}
OUTPUT_FIELDS = {"idx", "ans_qa_words", "ans_qa_sents", "choose_id"}
STANDARD_CHOOSE_KEYS = {"A", "B", "C", "D"}
SENTENCE_FINAL_PUNCTUATION = "。！？!?；;，,.．"


def unique_preserve_order(items: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = _clean_str(item)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def normalize_input(raw: Mapping[str, Any]) -> dict:
    idx = _first_present(raw, ("idx", "index", "id"))
    content = _first_present(raw, ("content", "poem", "text"))
    qa_words = _first_present(raw, ("qa_words", "words", "target_words"))
    qa_sents = _first_present(raw, ("qa_sents", "sentences", "target_sents"))
    choose = _first_present(raw, ("choose", "options", "emotion_options"))

    return {
        "idx": idx,
        "title": _clean_str(raw.get("title")),
        "author": _clean_str(raw.get("author")),
        "content": _clean_str(content),
        "qa_words": _normalize_text_list(qa_words),
        "qa_sents": _normalize_text_list(qa_sents),
        "choose": _normalize_choose(choose),
    }


def validate_input(sample: Mapping[str, Any]) -> dict:
    normalized = normalize_input(sample)
    errors: list[str] = []
    warnings: list[str] = []

    if normalized["idx"] is None:
        errors.append("invalid_input_missing_idx")
    if not normalized["content"]:
        errors.append("invalid_input_missing_content")
    if not normalized["qa_words"]:
        warnings.append("empty_qa_words")
    if not normalized["qa_sents"]:
        warnings.append("empty_qa_sents")
    if not normalized["choose"]:
        warnings.append("missing_choose")

    if len(unique_preserve_order(normalized["qa_words"])) < len(normalized["qa_words"]):
        warnings.append("duplicate_qa_words")
    if len(unique_preserve_order(normalized["qa_sents"])) < len(normalized["qa_sents"]):
        warnings.append("duplicate_qa_sents")
    if any(word == "" for word in normalized["qa_words"]):
        warnings.append("empty_qa_word")
    if any(sent == "" for sent in normalized["qa_sents"]):
        warnings.append("empty_qa_sent")

    choose_keys = set(normalized["choose"])
    if choose_keys and not choose_keys <= STANDARD_CHOOSE_KEYS:
        warnings.append("non_standard_choose_keys")
    if any(value == "" for value in normalized["choose"].values()):
        warnings.append("empty_choose_text")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized,
    }


def validate_output(output: Mapping[str, Any], task: Mapping[str, Any]) -> dict:
    normalized_task = normalize_input(task)
    errors: list[str] = []
    warnings: list[str] = []
    normalized: dict[str, Any] = {
        "idx": output.get("idx"),
        "ans_qa_words": _normalize_answer_mapping(output.get("ans_qa_words")),
        "ans_qa_sents": _normalize_answer_mapping(output.get("ans_qa_sents")),
        "choose_id": _clean_str(output.get("choose_id")),
    }

    output_keys = set(output)
    extra_fields = output_keys - OUTPUT_FIELDS
    missing_fields = OUTPUT_FIELDS - output_keys
    if extra_fields:
        errors.append("unexpected_output_fields")
    if missing_fields:
        errors.append("missing_output_fields")

    if output.get("idx") != normalized_task["idx"]:
        errors.append("idx_mismatch")
    if not isinstance(output.get("ans_qa_words"), Mapping):
        errors.append("invalid_ans_qa_words")
    if not isinstance(output.get("ans_qa_sents"), Mapping):
        errors.append("invalid_ans_qa_sents")
    if "choose_id" in output and not isinstance(output.get("choose_id"), str):
        errors.append("invalid_choose_id_type")

    _check_answer_coverage(
        normalized["ans_qa_words"],
        unique_preserve_order(normalized_task["qa_words"]),
        "ans_qa_words",
        errors,
        warnings,
    )
    _check_answer_coverage(
        normalized["ans_qa_sents"],
        unique_preserve_order(normalized_task["qa_sents"]),
        "ans_qa_sents",
        errors,
        warnings,
    )

    choose = normalized_task["choose"]
    choose_id = normalized["choose_id"]
    if choose:
        if choose_id not in choose:
            errors.append("invalid_choose_id")
    elif choose_id != "":
        errors.append("invalid_choose_id")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized,
    }


def validate_output_punctuation_normalized(
    output: Mapping[str, Any], task: Mapping[str, Any]
) -> dict:
    result = validate_output(output, task)
    normalized_task = normalize_input(task)
    required_sents = unique_preserve_order(normalized_task["qa_sents"])
    answer_sents = result["normalized"]["ans_qa_sents"]

    has_sentence_coverage_error = (
        "missing_ans_qa_sents" in result["errors"]
        or "unexpected_ans_qa_sents" in result["errors"]
    )
    if has_sentence_coverage_error and _has_punctuation_normalized_sentence_match(
        answer_sents,
        required_sents,
    ):
        result["errors"] = [
            error
            for error in result["errors"]
            if error not in {"missing_ans_qa_sents", "unexpected_ans_qa_sents"}
        ]
        if "punctuation_key_mismatch" not in result["warnings"]:
            result["warnings"].append("punctuation_key_mismatch")
        if not _has_empty_punctuation_normalized_sentence_answer(
            answer_sents,
            required_sents,
        ):
            result["warnings"] = [
                warning
                for warning in result["warnings"]
                if warning != "empty_ans_qa_sents"
            ]
        result["valid"] = not result["errors"]

    return result


def build_output_from_training(
    raw: Mapping[str, Any], task: Mapping[str, Any] | None = None
) -> tuple[dict | None, list[str]]:
    issues: list[str] = []
    source_task = normalize_input(task or raw)
    output: dict[str, Any] = {
        "idx": source_task["idx"],
        "ans_qa_words": {},
        "ans_qa_sents": {},
        "choose_id": "",
    }
    has_direct_label = False

    keywords = raw.get("keywords")
    if isinstance(keywords, Mapping):
        qa_words = unique_preserve_order(source_task["qa_words"])
        if not qa_words:
            qa_words = unique_preserve_order(keywords.keys())
            if qa_words:
                issues.append("qa_words_from_keywords")
        output["ans_qa_words"] = {
            word: _clean_str(keywords.get(word))
            for word in qa_words
        }
        if qa_words:
            has_direct_label = True
        if any(word not in keywords for word in qa_words):
            issues.append("missing_keyword_label")

    if raw.get("trans") not in (None, ""):
        issues.append("unmapped_trans")
    if raw.get("emotion") not in (None, ""):
        issues.append("unmapped_emotion")
    if raw.get("choose_id") not in (None, ""):
        issues.append("unmapped_choose_id")

    if not has_direct_label:
        return None, issues
    return output, issues


def _first_present(raw: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in raw:
            return raw[name]
    return None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [_clean_str(item) for item in value.keys()]
    if isinstance(value, str):
        return [_clean_str(value)] if value.strip() else []
    if isinstance(value, Iterable):
        return [_clean_str(item) for item in value]
    return [_clean_str(value)]


def _normalize_choose(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {_clean_str(key): _clean_str(option) for key, option in value.items()}
    if isinstance(value, str):
        return {}
    if isinstance(value, Iterable):
        return {_option_key(index): _clean_str(option) for index, option in enumerate(value)}
    return {}


def _normalize_answer_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {_clean_str(key): _clean_str(answer) for key, answer in value.items()}


def _option_key(index: int) -> str:
    key = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        key = chr(ord("A") + remainder) + key
    return key


def _check_answer_coverage(
    answers: Mapping[str, str],
    required_keys: list[str],
    field_name: str,
    errors: list[str],
    warnings: list[str],
) -> None:
    answer_keys = set(answers)
    required = set(required_keys)
    if required - answer_keys:
        errors.append(f"missing_{field_name}")
    if answer_keys - required:
        errors.append(f"unexpected_{field_name}")
    if any(answers.get(key, "") == "" for key in required):
        warnings.append(f"empty_{field_name}")


def _has_punctuation_normalized_sentence_match(
    answers: Mapping[str, str],
    required_keys: list[str],
) -> bool:
    answer_keys = set(answers)
    required = set(required_keys)
    if answer_keys == required:
        return False
    if len(answer_keys) != len(required):
        return False
    normalized_answer_keys = {
        _strip_sentence_final_punctuation(key)
        for key in answer_keys
    }
    normalized_required_keys = {
        _strip_sentence_final_punctuation(key)
        for key in required
    }
    if len(normalized_answer_keys) != len(answer_keys):
        return False
    if len(normalized_required_keys) != len(required):
        return False
    return normalized_answer_keys == normalized_required_keys


def _strip_sentence_final_punctuation(value: str) -> str:
    return value.rstrip(SENTENCE_FINAL_PUNCTUATION)


def _has_empty_punctuation_normalized_sentence_answer(
    answers: Mapping[str, str],
    required_keys: list[str],
) -> bool:
    answer_by_normalized_key = {
        _strip_sentence_final_punctuation(key): answer
        for key, answer in answers.items()
    }
    return any(
        answer_by_normalized_key.get(_strip_sentence_final_punctuation(key), "") == ""
        for key in required_keys
    )
