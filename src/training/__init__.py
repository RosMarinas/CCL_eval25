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

ZERO_SHOT_PROMPT = """你需要完成古诗词理解任务。请根据输入诗歌、目标词语、目标句子和情感选项，直接生成最终答案 JSON。

输出要求：
- 只输出一个合法 JSON 对象。
- 不要输出 Markdown 代码块。
- 不要输出解释、分析、证据、草稿或任何 JSON 之外的文字。
- JSON 字段必须且只能包含：idx、ans_qa_words、ans_qa_sents、choose_id。
- idx 必须与输入 idx 完全一致。
- ans_qa_words 是对象，key 必须逐字复制输入数组中的原始字符串，使用 qa_words 中的原词；包括标点、空格和全半角字符，不能删改句末标点；重复词语只输出一个 key；value 是该词在诗中的简洁解释。
- ans_qa_sents 是对象，key 必须逐字复制输入数组中的原始字符串，使用 qa_sents 中的原句；包括标点、空格和全半角字符，不能删改句末标点；重复句子只输出一个 key；value 是该句的简洁现代汉语翻译。
- choose_id 必须从 choose 的选项 ID 中选择一个最符合全诗情感的选项。

输入：
{input_json}

现在只输出最终 JSON："""

EVIDENCE_DRAFT_PROMPT = """你需要完成古诗词理解任务。请根据输入诗歌、目标词语、目标句子和情感选项，生成分析证据、情感分析和草稿答案。

输出要求：
- 只输出一个合法 JSON 对象，包含 evidence、sentiment 和 draft_answer 三个字段。
- evidence 包含 words（词语解释对象）、sentences（句子翻译对象）和 emotion（情感分析列表）三个子字段。
- sentiment 包含 primary（主要情感标签，必须使用受控词汇表中的标签）、secondary（次要情感标签列表）和 rationale（简短理由，不超过80字）。
- draft_answer 包含 ans_qa_words 和 ans_qa_sents，不包含 choose_id。
- 不要输出 Markdown 代码块或任何额外文字。

输入：
{input_json}

现在输出 evidence、sentiment 和 draft_answer："""

CRITIQUE_CORRECTION_PROMPT = """你需要对古诗词理解答案进行评审和修正。请根据输入诗歌、目标词语、目标句子、情感选项以及一个候选错误答案，生成评审意见和修正后的答案。

输出要求：
- 只输出一个合法 JSON 对象，包含 critique、correction_evidence 和 corrected_answer 三个字段。
- critique 包含 word_errors（词义错误列表）、sentence_errors（句译错误列表）和 emotion_error（情感分析错误对象）。
- emotion_error 包含 issue（问题描述）、expected_primary（正确情感标签）和 rationale_mismatch（理由）。
- correction_evidence 包含修正后的 words（词语解释）、sentences（句子翻译）和 emotion（情感分析）。
- corrected_answer 是可直接评测的最终答案，必须包含 idx、ans_qa_words、ans_qa_sents 和 choose_id。
- 不要输出 Markdown 代码块或任何额外文字。

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
