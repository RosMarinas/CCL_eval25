"""Shared utilities for training, evaluation, and prompt rendering.

Extracted from src/cli/train_b8.py and src/cli/train_bc8.py to eliminate
duplication across CLI scripts.  All training and eval scripts import
from here instead of from each other.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.eval import parse_json_object
from src.schema import unique_preserve_order

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ZERO_SHOT_PROMPT = """你是一名经验丰富古诗词理解评测教授。请根据输入诗歌、目标词语、目标句子和情感选项，生成可直接评测的最终答案 JSON。

请在内部完成：

1. 解释 qa_words 中每个词语在诗中的含义。
2. 翻译 qa_sents 中每个句子为现代汉语。
3. 从 choose 中选择最符合全诗主要情感的选项 ID。

输出要求：

* 只输出一个合法 JSON 对象，不要 Markdown，不要解释，不要 JSON 之外的文字。
* JSON 字段必须且只能包含：idx、ans_qa_words、ans_qa_sents、choose_id。
* idx 必须与输入 idx 完全一致。
* ans_qa_words 是对象：key 必须逐字复制 qa_words 中的原词；value 是简洁词义。
* ans_qa_sents 是对象：key 必须逐字复制 qa_sents 中的原句；value 是简洁现代汉语翻译。
* choose_id 必须是 choose 中已有的选项 ID，只输出选项字母。
* 所有字段都必须存在，不能为 null，不能输出空字符串。
* 若不确定，给出最可能答案，不要留空。

输入：
{input_json}

现在只输出最终 JSON："""

EVIDENCE_DRAFT_PROMPT = """你是一名经验丰富古诗词理解评测教授。请根据输入诗歌、目标词语、目标句子和情感选项，生成结构化证据、情感判断和草稿答案。

输出要求：

* 只输出一个合法 JSON 对象，不要 Markdown，不要 JSON 之外的文字。
* 顶层字段必须且只能包含：evidence、sentiment、draft_answer。
* qa_words 和 qa_sents 相关对象的 key 必须逐字复制输入中的原词和原句。
* 所有字段都必须存在，不能为 null。

JSON 结构：
{
"evidence": {
"words": {"原词": "简洁词义"},
"sentences": {"原句": "现代汉语翻译"},
"emotion": ["简短情感证据"]
},
"sentiment": {
"primary": "主要情感标签",
"secondary": ["次要情感标签"],
"rationale": "不超过80字的理由"
},
"draft_answer": {
"ans_qa_words": {"原词": "简洁词义"},
"ans_qa_sents": {"原句": "现代汉语翻译"}
}
}

primary 必须从以下标签中选择一个：
惜别感伤、送别不舍、离别愁绪、思乡怀远、羁旅思归、故园之思、
忧国伤时、报国壮志、兴亡之叹、山水闲适、田园之乐、隐逸情怀、
怀古伤今、历史沧桑、昔盛今衰、相思闺怨、爱情甜蜜、相思之苦、
人生无常、时光易逝、仕途失意、边塞征战、将士艰辛、厌战思归、其他

输入：
{input_json}

现在输出 evidence、sentiment 和 draft_answer："""

CRITIQUE_CORRECTION_PROMPT = """你是一名经验丰富古诗词理解评测教授。请根据输入诗歌、目标词语、目标句子、情感选项和候选答案，评审候选答案并生成修正后的最终答案。

评审原则：

* 候选答案语义正确但表达不同，不要误判为错。
* 只有明显误解词义、句意或全诗情感时才修正。
* 不确定时尽量保留候选答案中合理的部分。
* 若候选答案格式错误、缺字段或 key 未精确复制输入，必须修正。

输出要求：

* 只输出一个合法 JSON 对象，不要 Markdown，不要 JSON 之外的文字。
* 顶层字段必须且只能包含：critique、correction_evidence、corrected_answer。
* 所有字段都必须存在，不能为 null。
* corrected_answer 必须可直接评测，且字段必须且只能包含：idx、ans_qa_words、ans_qa_sents、choose_id。
* idx 必须与输入 idx 完全一致。
* ans_qa_words 的 key 必须逐字复制 qa_words 中的原词。
* ans_qa_sents 的 key 必须逐字复制 qa_sents 中的原句。
* choose_id 必须来自输入 choose 的选项 ID。

JSON 结构：
{
"critique": {
"word_errors": [],
"sentence_errors": [],
"emotion_error": {
"issue": "问题描述或无明显错误",
"expected_primary": "正确或最接近的主要情感",
"rationale_mismatch": "不匹配原因或无明显不匹配"
}
},
"correction_evidence": {
"words": {"原词": "修正后词义"},
"sentences": {"原句": "修正后译文"},
"emotion": {
"primary": "主要情感",
"rationale": "简短理由",
"selected_choose_id": "选项ID"
}
},
"corrected_answer": {
"idx": 输入idx,
"ans_qa_words": {"原词": "最终词义"},
"ans_qa_sents": {"原句": "最终译文"},
"choose_id": "选项ID"
}
}

输入：
{input_json}

候选答案：
{candidate_json}

现在输出评审和修正 JSON："""


TOP_FIELDS = frozenset({"idx", "ans_qa_words", "ans_qa_sents", "choose_id"})

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_train_source(source_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Load original training data from the train-data/ directory tree.

    Each JSON file is an array of items with keys: title, content, keywords,
    trans, emotion.  Sequential idx (0, 1, 2, ...) is assigned in file-sorted
    order.

    Returns a dict mapping idx -> {title, content}.  Author is always empty
    because the raw training data has no author field.
    """
    source_dir = Path(source_dir)
    samples: list[dict[str, Any]] = []
    idx = 0
    for json_path in sorted(source_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            samples.append({
                "idx": idx,
                "title": (item.get("title") or ""),
                "content": (item.get("content") or ""),
            })
            idx += 1

    logger.info("Loaded %d source samples from %s", len(samples), source_dir)
    return {s["idx"]: s for s in samples}


def load_answer_only(path: str | Path) -> list[dict[str, Any]]:
    """Load b8-answer-only.jsonl where each line is a final JSON record.

    Expected record format:
      {"idx": 0, "ans_qa_words": {...}, "ans_qa_sents": {...}, "choose_id": ""}
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info("Loaded %d answer-only records from %s", len(records), path)
    return records


def load_dev_data(path: str | Path) -> list[dict[str, Any]]:
    """Load dev split data (eval50.json), an array of task JSON objects."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data).__name__}")
    logger.info("Loaded %d dev samples from %s", len(data), path)
    return data


# ---------------------------------------------------------------------------
# Task construction and prompt rendering
# ---------------------------------------------------------------------------


def build_task_json(
    idx: int,
    title: str = "",
    author: str = "",
    content: str = "",
    qa_words: list[str] | None = None,
    qa_sents: list[str] | None = None,
    choose: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the unified input task JSON per docs/spec/data-schema.md Section 1."""
    return {
        "idx": idx,
        "title": title,
        "author": author,
        "content": content,
        "qa_words": qa_words or [],
        "qa_sents": qa_sents or [],
        "choose": choose or {},
    }


def render_prompt_text(task_json: dict[str, Any]) -> str:
    """Fill the zero-shot prompt template with the task JSON."""
    return ZERO_SHOT_PROMPT.format(
        input_json=json.dumps(task_json, ensure_ascii=False)
    )


def render_evidence_draft_text(task_json: dict[str, Any]) -> str:
    """Fill the evidence-draft prompt template with the task JSON."""
    return EVIDENCE_DRAFT_PROMPT.format(
        input_json=json.dumps(task_json, ensure_ascii=False)
    )


def render_critique_correction_text(
    task_json: dict[str, Any],
    candidate_json: dict[str, Any] | str,
) -> str:
    """Fill the critique-correction prompt template with task and candidate."""
    if isinstance(candidate_json, dict):
        candidate_str = json.dumps(candidate_json, ensure_ascii=False)
    else:
        candidate_str = str(candidate_json)
    return CRITIQUE_CORRECTION_PROMPT.format(
        input_json=json.dumps(task_json, ensure_ascii=False),
        candidate_json=candidate_str,
    )


def render_prompt_for_target(
    task_json: dict[str, Any],
    target: str,
    candidate: dict[str, Any] | None = None,
) -> str:
    """Render the appropriate prompt for the given target type."""
    if target == "final_json":
        return render_prompt_text(task_json)
    elif target == "evidence_draft":
        return render_evidence_draft_text(task_json)
    elif target == "critique_correction":
        return render_critique_correction_text(task_json, candidate or {})
    else:
        raise ValueError(f"Unknown target: {target}")


def format_user_prompt(task: dict[str, Any]) -> str:
    """Return the plain-text prompt (no chat template), matching eval format."""
    return render_prompt_text(task)


# ---------------------------------------------------------------------------
# Training data assembly
# ---------------------------------------------------------------------------


def build_training_pairs(
    answer_records: list[dict[str, Any]],
    source_index: dict[int, dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair each answer record with its reconstructed task JSON.

    For records lacking a source entry (should not happen with real data),
    missing fields default to empty strings and zero-length arrays.
    """
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rec in answer_records:
        idx = rec["idx"]
        source = source_index.get(idx, {})

        # qa_words / qa_sents are derived from the answer keys themselves
        qa_words = list(rec.get("ans_qa_words", {}).keys())
        qa_sents = list(rec.get("ans_qa_sents", {}).keys())

        task = build_task_json(
            idx=idx,
            title=source.get("title", ""),
            content=source.get("content", ""),
            qa_words=qa_words,
            qa_sents=qa_sents,
            choose={},
        )
        pairs.append((task, rec))

    logger.info("Built %d training pairs", len(pairs))
    return pairs


def prepare_dataset(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenise and produce input_ids + labels with prompt masking.

    Labels for the prompt part are set to -100 so the language-model loss
    is computed only on the assistant response.
    """
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    truncated_count = 0

    for task, output_json in pairs:
        prompt_text = render_prompt_text(task)
        response_text = json.dumps(output_json, ensure_ascii=False)
        eos = tokenizer.eos_token or ""
        full_text = prompt_text + response_text + eos

        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )["input_ids"]

        prompt_ids = tokenizer(
            prompt_text,
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = len(prompt_ids)

        if len(full_ids) >= max_length:
            truncated_count += 1

        labels = ([-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:])

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

        while len(labels) < len(full_ids):
            labels.append(-100)

        all_input_ids.append(full_ids)
        all_labels.append(labels)
        all_attention_masks.append([1] * len(full_ids))

    if truncated_count:
        logger.warning(
            "%d / %d samples exceeded max_length=%d and were truncated",
            truncated_count,
            len(pairs),
            max_length,
        )

    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    })
    logger.info("Prepared dataset with %d samples", len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Model loading with QLoRA
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class PadCollator:
    """Pads a batch of tokenised sequences (with labels) to equal length.

    Uses torch.nn.utils.rnn.pad_sequence directly instead of tokenizer.pad
    to avoid version-specific padding API changes in transformers >= 5.x.
    """

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------------
# Dev-set evaluation
# ---------------------------------------------------------------------------


def classify_json_errors(
    parsed: dict[str, Any] | None,
    task: dict[str, Any],
) -> list[str]:
    """Classify JSON/schema errors against the expected unified input schema.

    Returns an ordered list of error category strings (empty = valid).
    """
    if parsed is None:
        return ["parse_error"]

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

    target_words = unique_preserve_order(task.get("qa_words", []))
    target_sents = unique_preserve_order(task.get("qa_sents", []))

    if isinstance(word_answers, dict):
        word_keys = {str(k) for k in word_answers}
        if {str(k) for k in target_words} - word_keys:
            errors.append("missing_word_key")

    if isinstance(sent_answers, dict):
        sent_keys = {str(k) for k in sent_answers}
        if {str(k) for k in target_sents} - sent_keys:
            errors.append("missing_sentence_key")

    if isinstance(word_answers, dict):
        for v in word_answers.values():
            if not isinstance(v, str) or not v.strip():
                errors.append("empty_required_answer")
                break

    if isinstance(sent_answers, dict):
        for v in sent_answers.values():
            if not isinstance(v, str) or not v.strip():
                errors.append("empty_required_answer")
                break

    choices = task.get("choose") or {}
    if isinstance(choose_id, str):
        if choices and choose_id not in {str(k) for k in choices}:
            errors.append("invalid_choose_id")
        if not choices and choose_id != "":
            errors.append("invalid_choose_id")

    return errors


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dev_tasks: list[dict[str, Any]],
    max_new_tokens: int = 1024,
) -> dict[str, float | int]:
    """Run the model on dev tasks and compute JSON error rate.

    Core errors (parse_error, missing_top_field, idx_mismatch,
    wrong_field_type, missing_word_key, missing_sentence_key,
    empty_required_answer, invalid_choose_id) count as failures.
    """
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()

    total = len(dev_tasks)
    json_errors = 0

    for task in dev_tasks:
        prompt = render_prompt_text(task)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed, _parse_errs = parse_json_object(raw)
        error_categories = classify_json_errors(parsed, task)
        core = {
            "parse_error",
            "missing_top_field",
            "idx_mismatch",
            "wrong_field_type",
            "missing_word_key",
            "missing_sentence_key",
            "empty_required_answer",
            "invalid_choose_id",
        }
        if set(error_categories) & core:
            json_errors += 1

    error_rate = json_errors / total if total else 0.0
    logger.info(
        "Evaluation: %d/%d JSON errors (%.1f%%)",
        json_errors, total, error_rate * 100,
    )
    return {"json_errors": json_errors, "total": total, "json_error_rate": error_rate}
