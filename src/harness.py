"""Local reasoner-to-formatter harness utilities."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

try:
    from src.schema import unique_preserve_order as _schema_unique_preserve_order
except ImportError:
    _schema_unique_preserve_order = None


Report = dict[str, Any]
FormatterFn = Callable[[dict[str, Any]], str | dict[str, Any]]

FINAL_FIELDS = {"idx", "ans_qa_words", "ans_qa_sents", "choose_id"}
WORD_LIMIT = 40
SENTENCE_LIMIT = 80


def unique_preserve_order(items: list[Any] | None) -> list[Any]:
    if _schema_unique_preserve_order is not None:
        return _schema_unique_preserve_order(items or [])

    seen = set()
    unique_items = []
    for item in items or []:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def parse_reasoner_output(raw_output: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse a JSON object, allowing markdown fences or small text wrappers."""
    if isinstance(raw_output, dict):
        return raw_output
    if raw_output is None:
        return None
    if not isinstance(raw_output, str):
        return None

    text = raw_output.strip()
    if not text:
        return None

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    parsed = _loads_object(text)
    if parsed is not None:
        return parsed

    for candidate in _json_object_candidates(text):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


def validate_reasoner_output(task: dict[str, Any], reasoner_output: dict[str, Any] | None) -> Report:
    report = _empty_report(valid_json=reasoner_output is not None)
    if reasoner_output is None:
        report["missing_fields"].extend(["idx", "evidence", "draft_answer"])
        return report

    _validate_common_structure(task, reasoner_output, report)
    return report


def should_skip_formatter(report: Report) -> bool:
    return (
        report.get("valid_json") is True
        and not report.get("missing_fields")
        and not report.get("missing_words")
        and not report.get("missing_sentences")
        and not report.get("invalid_choose_id")
        and not report.get("overlong_fields")
        and not report.get("suspected_conflicts")
    )


def build_formatter_input(
    task: dict[str, Any],
    reasoner_output: dict[str, Any] | None,
    report: Report,
) -> dict[str, Any]:
    return {
        "task": task,
        "reasoner_output": reasoner_output or {},
        "validator_report": report,
    }


def build_formatter_prompt(formatter_input: dict[str, Any]) -> str:
    payload = json.dumps(formatter_input, ensure_ascii=False, indent=2)
    return (
        "你是古诗词理解任务的 formatter / verifier。\n\n"
        "输入包括原题 task、reasoner_output 和 validator_report。\n"
        "你的任务是把 reasoner 的 draft_answer 整理成最终提交 JSON，并做轻量校验。\n\n"
        "必须遵守：\n"
        "1. 默认相信 reasoner 的 draft_answer，不要重新做题。\n"
        "2. 不要输出推理过程、解释、Markdown 或代码块，只输出一个 JSON 对象。\n"
        "3. 最终 JSON 只能包含 idx、ans_qa_words、ans_qa_sents、choose_id 四个字段。\n"
        "4. idx 必须使用 task.idx。\n"
        "5. ans_qa_words 的 key 必须覆盖 task.qa_words 去重后的词语。\n"
        "6. ans_qa_sents 的 key 必须覆盖 task.qa_sents 去重后的句子。\n"
        "7. choose_id 必须来自 task.choose 的选项 ID；标准样本应为 A、B、C、D 之一。\n"
        "8. 答案应简洁：词义答案通常不超过 40 个中文字符，句子翻译通常不超过 80 个中文字符。\n"
        "9. 只在 JSON 不合法、缺项、choose_id 非法、答案明显过长或存在明显直接冲突时轻微修改。\n"
        "10. 如果缺项无法根据 draft_answer 和 evidence 补齐，使用最短占位式答案，不要发明长解释。\n\n"
        "输出最终 JSON。\n\n"
        f"{payload}"
    )


def postprocess_draft(task: dict[str, Any], reasoner_output: dict[str, Any]) -> dict[str, Any]:
    draft = _draft(reasoner_output)
    return {
        "idx": task.get("idx"),
        "ans_qa_words": dict(draft.get("ans_qa_words") or {}),
        "ans_qa_sents": dict(draft.get("ans_qa_sents") or {}),
        "choose_id": _normalize_choose_id(draft.get("choose_id"), task.get("choose") or {}),
    }


def fallback_final(
    task: dict[str, Any],
    reasoner_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    draft = _draft(reasoner_output or {})
    draft_words = draft.get("ans_qa_words") if isinstance(draft.get("ans_qa_words"), dict) else {}
    draft_sents = draft.get("ans_qa_sents") if isinstance(draft.get("ans_qa_sents"), dict) else {}

    final_words = {}
    for word in unique_preserve_order(task.get("qa_words") or []):
        value = draft_words.get(word, "")
        final_words[word] = value if isinstance(value, str) else ""

    final_sents = {}
    for sent in unique_preserve_order(task.get("qa_sents") or []):
        value = draft_sents.get(sent, "")
        final_sents[sent] = value if isinstance(value, str) else ""

    choose_id = _normalize_choose_id(draft.get("choose_id"), task.get("choose") or {})
    return {
        "idx": task.get("idx"),
        "ans_qa_words": final_words,
        "ans_qa_sents": final_sents,
        "choose_id": choose_id,
    }


def decide_next_action(report: Report) -> str:
    if not report.get("valid_json"):
        return "retry_reasoner"

    missing_fields = set(report.get("missing_fields") or [])
    if "draft_answer" in missing_fields or "draft_answer.choose_id" in missing_fields:
        return "retry_reasoner"
    if report.get("invalid_choose_id"):
        return "retry_reasoner"
    if should_skip_formatter(report):
        return "use_draft"
    if (
        report.get("missing_fields")
        or report.get("missing_words")
        or report.get("missing_sentences")
        or report.get("overlong_fields")
        or report.get("suspected_conflicts")
    ):
        return "call_formatter"
    return "fallback"


def run_harness_once(
    task: dict[str, Any],
    raw_reasoner_output: str | dict[str, Any] | None,
    formatter_fn: FormatterFn | None = None,
) -> dict[str, Any]:
    reasoner_output = parse_reasoner_output(raw_reasoner_output)
    report = validate_reasoner_output(task, reasoner_output)
    action = decide_next_action(report)
    formatter_called = False
    fallback_used = False
    reasoner_retried = action == "retry_reasoner"

    if action == "use_draft" and reasoner_output is not None:
        final_answer = postprocess_draft(task, reasoner_output)
    elif formatter_fn is not None and reasoner_output is not None:
        formatter_called = True
        formatter_input = build_formatter_input(task, reasoner_output, report)
        formatter_output = parse_reasoner_output(formatter_fn(formatter_input))
        final_report = validate_final_output(task, formatter_output)
        if should_skip_formatter(final_report):
            final_answer = formatter_output
        else:
            fallback_used = True
            final_answer = fallback_final(task, reasoner_output)
    else:
        fallback_used = True
        final_answer = fallback_final(task, reasoner_output)

    final_report = validate_final_output(task, final_answer)
    if not should_skip_formatter(final_report):
        fallback_used = True
        final_answer = fallback_final(task, reasoner_output)
        final_report = validate_final_output(task, final_answer)

    return {
        "idx": task.get("idx"),
        "action": action,
        "reasoner_retried": reasoner_retried,
        "formatter_called": formatter_called,
        "fallback_used": fallback_used,
        "validator_report": report,
        "final_validator_report": final_report,
        "reasoner_output": reasoner_output,
        "final_answer": final_answer,
    }


def validate_final_output(task: dict[str, Any], final_output: dict[str, Any] | None) -> Report:
    report = _empty_report(valid_json=final_output is not None)
    if final_output is None:
        report["missing_fields"].extend(["idx", "ans_qa_words", "ans_qa_sents", "choose_id"])
        return report

    extra_fields = sorted(set(final_output) - FINAL_FIELDS)
    if extra_fields:
        report["suspected_conflicts"].append("extra_top_field:" + ",".join(extra_fields))
    _validate_final_fields(task, final_output, report, prefix="")
    return report


def _empty_report(valid_json: bool) -> Report:
    return {
        "valid_json": valid_json,
        "missing_fields": [],
        "missing_words": [],
        "missing_sentences": [],
        "invalid_choose_id": False,
        "overlong_fields": [],
        "suspected_conflicts": [],
    }


def _validate_common_structure(
    task: dict[str, Any],
    output: dict[str, Any],
    report: Report,
) -> None:
    for field in ("idx", "evidence", "draft_answer"):
        if field not in output:
            report["missing_fields"].append(field)

    if output.get("idx") != task.get("idx"):
        report["suspected_conflicts"].append("idx_mismatch")

    draft = output.get("draft_answer")
    if not isinstance(draft, dict):
        return

    _validate_final_fields(task, draft, report, prefix="draft_answer.")
    _detect_evidence_choose_conflict(output, draft, task, report)


def _validate_final_fields(
    task: dict[str, Any],
    answer: dict[str, Any],
    report: Report,
    prefix: str,
) -> None:
    if not prefix and "idx" not in answer:
        report["missing_fields"].append("idx")

    for field in ("ans_qa_words", "ans_qa_sents", "choose_id"):
        if field not in answer:
            report["missing_fields"].append(prefix + field)

    if not prefix and answer.get("idx") != task.get("idx"):
        report["suspected_conflicts"].append("idx_mismatch")

    ans_words = answer.get("ans_qa_words")
    if not isinstance(ans_words, dict):
        report["missing_fields"].append(prefix + "ans_qa_words")
        ans_words = {}
    for word in unique_preserve_order(task.get("qa_words") or []):
        value = ans_words.get(word)
        if _is_empty_answer(value):
            report["missing_words"].append(word)
        elif isinstance(value, str) and len(value) > WORD_LIMIT:
            report["overlong_fields"].append(f"{prefix}ans_qa_words.{word}")

    ans_sents = answer.get("ans_qa_sents")
    if not isinstance(ans_sents, dict):
        report["missing_fields"].append(prefix + "ans_qa_sents")
        ans_sents = {}
    for sent in unique_preserve_order(task.get("qa_sents") or []):
        value = ans_sents.get(sent)
        if _is_empty_answer(value):
            report["missing_sentences"].append(sent)
        elif isinstance(value, str) and len(value) > SENTENCE_LIMIT:
            report["overlong_fields"].append(f"{prefix}ans_qa_sents.{sent}")

    choose = task.get("choose") or {}
    choose_id = _normalize_choose_id(answer.get("choose_id"), choose)
    if choose and choose_id not in choose:
        report["invalid_choose_id"] = True
    if not choose and choose_id != "":
        report["invalid_choose_id"] = True


def _draft(output: dict[str, Any]) -> dict[str, Any]:
    draft = output.get("draft_answer")
    return draft if isinstance(draft, dict) else {}


def _normalize_choose_id(value: Any, choose: dict[str, Any]) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    text = text.translate(str.maketrans("ＡＢＣＤＥＦＧＨＩＪ", "ABCDEFGHIJ"))
    text = re.sub(r"\s+", "", text)
    if text in choose:
        return text
    matches = [key for key in choose if key in text]
    return matches[0] if len(matches) == 1 else ""


def _detect_evidence_choose_conflict(
    output: dict[str, Any],
    draft: dict[str, Any],
    task: dict[str, Any],
    report: Report,
) -> None:
    choose = task.get("choose") or {}
    draft_choose = _normalize_choose_id(draft.get("choose_id"), choose)
    if not draft_choose:
        return

    evidence_text = json.dumps(output.get("evidence") or {}, ensure_ascii=False)
    mentioned = set()
    for choose_id in choose:
        if re.search(rf"(?:选|选择|应为|答案|choose_id)[：:\s\"']*{re.escape(choose_id)}", evidence_text, re.IGNORECASE):
            mentioned.add(choose_id)
    if mentioned and draft_choose not in mentioned:
        report["suspected_conflicts"].append("evidence_choose_conflict")


def _is_empty_answer(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, dict):
        return len(value) == 0
    return True


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_object_candidates(text: str) -> list[str]:
    candidates = []
    starts = [index for index, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : index + 1])
                    break
    return candidates
