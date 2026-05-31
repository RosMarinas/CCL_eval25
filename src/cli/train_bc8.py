#!/usr/bin/env python3
"""BC8 mixed-distillation QLoRA training for CCL25.

Initialises from the B8 LoRA checkpoint, then continues QLoRA on a
mixed-distillation dataset with three target types:

  - final_json (60%)        — answer-only final JSON output
  - evidence_draft (30%)    — evidence + sentiment + draft_answer
  - critique_correction (10%) — critique + correction_evidence + corrected_answer

Each epoch re-samples the training data to respect the 60/30/10 ratio.

Usage:
    python src/cli/train_bc8.py \
        --base-model Qwen/Qwen3-8B \
        --b8-checkpoint checkpoints/B8/adapter/ \
        --train-data data/training/bc8-mixed/train.jsonl \
        --dev-data data/splits/eval50.json \
        --output-dir checkpoints/BC8 \
        --lora-r 16 --lora-alpha 32 \
        --lr 5e-5 \
        --epochs 1 \
        --batch-size 4 --grad-accum 16 \
        --seq-length 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.eval import parse_json_object

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Prompt templates (one per target type)
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

# Schema: the only permitted top-level keys in the final JSON
TOP_FIELDS = frozenset({"idx", "ans_qa_words", "ans_qa_sents", "choose_id"})

# ---------------------------------------------------------------------------
# Source data loading (for reconstructing tasks from final_json records)
# ---------------------------------------------------------------------------


def load_train_source(source_dir: str | Path) -> dict[int, dict[str, Any]]:
    """Load original training data from the train-data/ directory tree.

    Each JSON file is an array of items with keys: title, content, keywords,
    trans, emotion.  Sequential idx is assigned in file-sorted order.

    Returns a dict mapping idx -> {title, content}.
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


# ---------------------------------------------------------------------------
# BC8 data loading
# ---------------------------------------------------------------------------


def load_bc8_data(path: str | Path) -> list[dict[str, Any]]:
    """Load BC8 mixed-distillation training data from JSONL."""
    path = Path(path)
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    logger.info("Loaded %d BC8 records from %s", len(records), path)
    return records


def report_target_distribution(records: list[dict[str, Any]]) -> None:
    """Log target distribution."""
    targets: Counter[str] = Counter(r.get("target", "unknown") for r in records)
    total = len(records)
    logger.info("Target distribution (%d total):", total)
    for target, count in targets.most_common():
        pct = count / total * 100
        logger.info("  %s: %d (%.1f%%)", target, count, pct)


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
# Task construction and rendering
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
    """Build the unified input task JSON per docs/contracts/data-schema.md Section 1."""
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


def unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate a list preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_task_and_output(
    record: dict[str, Any],
    source_index: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Extract (task_json, output_json, target) from a BC8 record.

    Handles two formats:
      1. Unified:  {target, task, output}
      2. Original (from build_training_data.py): target-specific fields
    """
    target = record.get("target", "unknown")

    # ---- Format 1: unified {target, task, output} ----
    if "task" in record and isinstance(record["task"], dict) and "output" in record:
        task = record["task"]
        output = record["output"]
        return task, output, target

    # ---- Format 2: original fields from build_training_data.py ----
    if target == "final_json":
        # Reconstruct task from source index + answer keys
        idx = record["idx"]
        src = source_index.get(idx, {})
        qa_words = list(record.get("ans_qa_words", {}).keys())
        qa_sents = list(record.get("ans_qa_sents", {}).keys())
        task = build_task_json(
            idx=idx,
            title=src.get("title", ""),
            content=src.get("content", ""),
            qa_words=qa_words,
            qa_sents=qa_sents,
            choose={},
        )
        output = {
            "idx": record.get("idx"),
            "ans_qa_words": record.get("ans_qa_words", {}),
            "ans_qa_sents": record.get("ans_qa_sents", {}),
            "choose_id": record.get("choose_id", ""),
        }
        return task, output, target

    elif target == "evidence_draft":
        # Teacher-data record has task key
        task = record.get("task", {})
        output = {
            "evidence": record.get("evidence", {}),
            "sentiment": record.get("sentiment", {}),
            "draft_answer": record.get("draft_answer", {}),
        }
        return task, output, target

    elif target == "critique_correction":
        # Teacher-critique record
        task = record.get("task", {})
        candidate = record.get("candidate_answer", record.get("task", {}))
        output = {
            "critique": record.get("critique", {}),
            "correction_evidence": record.get("correction_evidence", {}),
            "corrected_answer": record.get("corrected_answer", {}),
        }
        # Store candidate for prompt rendering
        return task, output, target, candidate

    # Last resort: if record has 'task' and 'output' at top level, use them
    if "task" in record:
        task = record["task"]
        output = {k: v for k, v in record.items() if k not in ("target", "task")}
        return task, output, target

    raise ValueError(f"Cannot extract task/output from record: target={target}, keys={list(record.keys())}")


# ---------------------------------------------------------------------------
# Chat-template formatting
# ---------------------------------------------------------------------------


def format_bc8_sample(
    task: dict[str, Any],
    output_json: dict[str, Any],
    target: str,
    tokenizer: AutoTokenizer,
    candidate_json: dict[str, Any] | None = None,
) -> str:
    """Format one BC8 training sample as a full chat conversation string.

    Uses plain text (no chat template) so training matches eval format.
    """
    prompt = render_prompt_for_target(task, target, candidate=candidate_json)
    response = json.dumps(output_json, ensure_ascii=False)
    eos = tokenizer.eos_token or ""
    return prompt + response + eos


def format_user_prompt(
    task: dict[str, Any],
    tokenizer: AutoTokenizer,
    target: str = "final_json",
    candidate_json: dict[str, Any] | None = None,
) -> str:
    """Return the plain-text prompt (no chat template), matching eval format."""
    return render_prompt_for_target(task, target, candidate=candidate_json)


# ---------------------------------------------------------------------------
# Weighted per-epoch sampling and dataset preparation
# ---------------------------------------------------------------------------


def create_weighted_samples(
    records: list[dict[str, Any]],
    source_index: dict[int, dict[str, Any]],
    ratio: dict[str, float],
    total_steps: int,
    effective_batch_size: int,
    tokenizer: AutoTokenizer,
    max_length: int,
    seed: int,
) -> Dataset:
    """Build a training dataset with per-batch weighted sampling.

    For each training step, samples are drawn from each target pool
    according to the specified ratio, ensuring every batch approximately
    respects the desired proportions.
    """
    # Group records by target
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_target[rec.get("target", "unknown")].append(rec)

    for t in ratio:
        if t not in by_target:
            logger.warning("Target %s has zero records in training data", t)
        else:
            logger.info("  %s: %d available", t, len(by_target[t]))

    rng = random.Random(seed)
    total_pairs: list[tuple[dict, dict, str, dict | None]] = []

    for step in range(total_steps):
        step_pairs: list[tuple[dict, dict, str, dict | None]] = []

        for target, proportion in ratio.items():
            pool = by_target.get(target, [])
            if not pool:
                continue
            # Number of samples for this target in this step
            n = max(1, int(effective_batch_size * proportion))
            n = min(n, len(pool))
            selected = rng.sample(pool, n)
            for rec in selected:
                try:
                    result = extract_task_and_output(rec, source_index)
                    if target == "critique_correction":
                        # Returns 4-tuple with candidate
                        task, output, tgt, candidate = result  # type: ignore
                    else:
                        task, output, tgt = result  # type: ignore
                        candidate = None
                    step_pairs.append((task, output, target, candidate))
                except (ValueError, KeyError) as e:
                    logger.warning("Skipping record: %s", e)
                    continue

        # Pad if we didn't get enough (unlikely with the max(1, ...) above)
        while len(step_pairs) < effective_batch_size and by_target:
            # Fill with random samples from any pool
            fallback_target = rng.choice(list(by_target.keys()))
            if by_target[fallback_target]:
                rec = rng.choice(by_target[fallback_target])
                try:
                    if fallback_target == "critique_correction":
                        t, o, tg, c = extract_task_and_output(rec, source_index)  # type: ignore
                        step_pairs.append((t, o, tg, c))
                    else:
                        t, o, tg = extract_task_and_output(rec, source_index)
                        step_pairs.append((t, o, tg, None))
                except (ValueError, KeyError):
                    pass

        step_pairs = step_pairs[:effective_batch_size]
        total_pairs.extend(step_pairs)

    logger.info("Generated %d training pairs across %d steps", len(total_pairs), total_steps)
    return tokenize_pairs(total_pairs, tokenizer, max_length)


def tokenize_pairs(
    pairs: list[tuple[dict, dict, str, dict | None]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenize formatted pairs into a HuggingFace Dataset.

    Uses prompt-masked labels (-100 for the user/prompt portion).
    """
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    truncated_count = 0
    total = len(pairs)

    for task, output, target, candidate in pairs:
        full_text = format_bc8_sample(task, output, target, tokenizer, candidate_json=candidate)
        prompt_text = format_user_prompt(task, tokenizer, target=target, candidate_json=candidate)

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

        # Labels: mask the prompt, keep the response
        labels = ([-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:])

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

        # Pad labels to match input length if prompt was truncated
        while len(labels) < len(full_ids):
            labels.append(-100)

        all_input_ids.append(full_ids)
        all_labels.append(labels)
        all_attention_masks.append([1] * len(full_ids))

    if truncated_count:
        logger.warning(
            "%d / %d samples exceeded max_length=%d and were truncated",
            truncated_count, total, max_length,
        )

    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks,
    })
    logger.info("Tokenized dataset: %d samples", len(dataset))
    return dataset


# ---------------------------------------------------------------------------
# Model loading with QLoRA + B8 adapter initialization
# ---------------------------------------------------------------------------


def load_quantized_model(
    model_name: str,
    device: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in 4-bit NF4 with double quant and bf16 compute dtype.

    Args:
        model_name: HuggingFace model ID.
        device: Device map (e.g. "auto", "cuda:0", or 0).
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model, tokenizer


def load_b8_adapter(
    model: AutoModelForCausalLM,
    b8_checkpoint: str,
) -> PeftModel:
    """Load the B8 LoRA adapter onto the quantized base model.

    The returned model is ready for continued training (the B8 adapter
    is set as trainable).
    """
    logger.info("Loading B8 adapter from %s", b8_checkpoint)
    model = PeftModel.from_pretrained(
        model,
        b8_checkpoint,
        is_trainable=True,
    )
    model.print_trainable_parameters()
    return model


def apply_lora(
    model: AutoModelForCausalLM,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> AutoModelForCausalLM:
    """Apply NEW LoRA adapters (used when NOT loading from B8 checkpoint)."""
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


class PadCollator:
    """Pads a batch of tokenised sequences (with labels) to equal length."""

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

    # Top-level field coverage
    missing = TOP_FIELDS - parsed.keys()
    if missing:
        errors.append("missing_top_field")

    # Extra fields
    if parsed.keys() - TOP_FIELDS:
        errors.append("extra_top_field")

    # idx fidelity
    if "idx" in parsed and parsed.get("idx") != task.get("idx"):
        errors.append("idx_mismatch")

    # Field type checks
    word_answers = parsed.get("ans_qa_words")
    sent_answers = parsed.get("ans_qa_sents")
    choose_id = parsed.get("choose_id")

    if "ans_qa_words" in parsed and not isinstance(word_answers, dict):
        errors.append("wrong_field_type")
    if "ans_qa_sents" in parsed and not isinstance(sent_answers, dict):
        errors.append("wrong_field_type")
    if "choose_id" in parsed and not isinstance(choose_id, str):
        errors.append("wrong_field_type")

    # Key coverage
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

    # Answer value checks
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

    # choose_id validity
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
    """Run the model on dev tasks and compute JSON error rate."""
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
        "Dev eval: %d / %d samples with JSON errors (rate = %.4f)",
        json_errors, total, error_rate,
    )
    return {"json_error_rate": error_rate, "error_count": json_errors, "total": total}


# ---------------------------------------------------------------------------
# Training ratio (50/25/25)
# ---------------------------------------------------------------------------

TARGET_RATIO: dict[str, float] = {
    "final_json": 0.5,
    "evidence_draft": 0.25,
    "critique_correction": 0.25,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def train_bc8(args: argparse.Namespace) -> int:
    """Execute the BC8 training pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    logger.info("Step 1/7: Loading BC8 training data ...")
    records = load_bc8_data(args.train_data)
    if not records:
        logger.error("No BC8 training records found. Aborting.")
        return 1

    report_target_distribution(records)

    # Load source index for final_json samples (needed for task reconstruction)
    source_index: dict[int, dict[str, Any]] = {}
    if args.train_source:
        logger.info("Loading source training data from %s", args.train_source)
        source_index = load_train_source(args.train_source)
    else:
        logger.info("No --train-source provided; final_json samples will have minimal task context")

    # ---- Step 2: Determine training steps with weighted ratio ----
    logger.info("Step 2/7: Computing weighted sampling plan ...")
    effective_batch = args.batch_size * args.grad_accum

    # Count per target to estimate steps
    target_counts: Counter[str] = Counter(r.get("target", "unknown") for r in records)
    available_per_target = {t: target_counts.get(t, 0) for t in TARGET_RATIO}

    # Total samples we can draw per epoch respecting the ratio
    # N * ratio[t] <= available_per_target[t] for all t
    max_per_target = {}
    for t, proportion in TARGET_RATIO.items():
        avail = available_per_target.get(t, 0)
        if avail > 0:
            max_per_target[t] = avail / proportion if proportion > 0 else float("inf")

    max_total = min(max_per_target.values()) if max_per_target else 0
    if max_total <= 0:
        logger.error("Cannot determine sample counts from available data.")
        return 1

    # Total training pairs per epoch (rounded down to effective_batch multiple)
    per_epoch_samples = int(max_total)
    per_epoch_samples = max(effective_batch, per_epoch_samples - (per_epoch_samples % effective_batch))

    steps_per_epoch = max(1, per_epoch_samples // effective_batch)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    total_steps = max(10, total_steps)  # Minimum meaningful training

    logging_steps = max(1, total_steps // 10) if total_steps > 1 else 1
    save_steps = max(1, total_steps)

    logger.info(
        "  per_epoch_samples=%d  effective_bs=%d  steps_per_epoch=%d  "
        "epochs=%.2f  total_steps=%d",
        per_epoch_samples, effective_batch, steps_per_epoch, args.epochs, total_steps,
    )

    # ---- Step 3: Load model + tokenizer with 4-bit quant + B8 adapter ----
    logger.info("Step 3/7: Loading base model with 4-bit quantization (device=%s) ...", args.device)
    model, tokenizer = load_quantized_model(args.base_model, device=args.device)

    logger.info("  Loading B8 adapter from %s ...", args.b8_checkpoint)
    model = load_b8_adapter(model, args.b8_checkpoint)
    if not args.no_gc:
        model.gradient_checkpointing_enable()
        logger.info("  Gradient checkpointing enabled")
    else:
        logger.info("  Gradient checkpointing disabled (--no-gc)")

    # ---- Step 4: Build weighted dataset ----
    logger.info("Step 4/7: Building weighted training dataset ...")
    dataset = create_weighted_samples(
        records=records,
        source_index=source_index,
        ratio=TARGET_RATIO,
        total_steps=total_steps,
        effective_batch_size=effective_batch,
        tokenizer=tokenizer,
        max_length=args.seq_length,
        seed=args.seed,
    )

    # ---- Step 5: Train ----
    logger.info("Step 5/7: Starting training ...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=total_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=args.weight_decay,
        bf16=True,
        tf32=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,
        report_to="none",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=not args.no_gc,
        optim="adamw_torch",
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer),
    )
    # transformers >= 5.x renamed tokenizer -> processing_class
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)

    train_result = trainer.train()
    final_loss = train_result.training_loss if hasattr(train_result, "training_loss") else None
    logger.info("Training complete. Final loss: %s", final_loss if final_loss else "N/A")

    # ---- Step 6: Save LoRA adapters ----
    logger.info("Step 6/7: Saving LoRA adapters ...")
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("  Adapters saved to %s", adapter_path)

    # ---- Step 7: Evaluate on dev set ----
    eval_results: dict[str, float | int] = {}
    if args.dev_data:
        logger.info("Step 7/7: Evaluating on dev set ...")
        if not args.no_gc:
            model.gradient_checkpointing_disable()
        dev_tasks = load_dev_data(args.dev_data)
        eval_results = evaluate(model, tokenizer, dev_tasks, args.eval_max_new_tokens)

        eval_path = output_dir / "eval_result.json"
        eval_path.write_text(
            json.dumps(eval_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("  Eval results written to %s", eval_path)

    # Save training metadata
    metadata = {
        "base_model": args.base_model,
        "b8_checkpoint": args.b8_checkpoint,
        "train_data": args.train_data,
        "dev_data": args.dev_data,
        "output_dir": args.output_dir,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "seq_length": args.seq_length,
        "effective_batch": effective_batch,
        "total_steps": total_steps,
        "ratio": dict(TARGET_RATIO),
        "available_per_target": dict(available_per_target),
        "final_loss": final_loss,
        "eval_results": eval_results,
    }
    meta_path = output_dir / "training_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("  Training metadata saved to %s", meta_path)

    logger.info("BC8 training complete.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC8 mixed-distillation QLoRA training for CCL25.",
    )

    # ---- Model ----
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--b8-checkpoint",
        required=True,
        help="Path to B8 LoRA adapter directory",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device map for model (default: "auto"). Use "cuda:0" for single GPU.',
    )
    parser.add_argument(
        "--no-gc",
        action="store_true",
        default=False,
        help="Disable gradient checkpointing (faster but uses more memory)",
    )

    # ---- Data ----
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to bc8-mixed/train.jsonl",
    )
    parser.add_argument(
        "--train-source",
        default=None,
        help="Path to train-data/ directory (for reconstructing final_json task context)",
    )
    parser.add_argument(
        "--dev-data",
        default=None,
        help="Path to dev split JSON (e.g. data/splits/eval50.json)",
    )

    # ---- Checkpoint output ----
    parser.add_argument(
        "--output-dir",
        default="checkpoints/BC8",
        help="Output directory (default: checkpoints/BC8)",
    )

    # ---- LoRA hyper-parameters ----
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")

    # ---- Optimisation ----
    parser.add_argument("--lr", type=float, default=5e-5, help="Peak learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4)")
    parser.add_argument(
        "--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048, help="Max sequence length (default: 2048)"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.03, help="Warmup ratio (default: 0.03)"
    )
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (default: 0.0)")

    # ---- Sampling ----
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for weighted sampling (default: 42)"
    )

    # ---- Evaluation ----
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=1024,
        help="Max generated tokens for dev eval (default: 1024)",
    )

    args = parser.parse_args(argv)

    # Resolve --train-source default relative to project root
    if args.train_source is None:
        resolved_source = PROJECT_ROOT / "data" / "train-data"
        if resolved_source.exists():
            args.train_source = str(resolved_source)
        else:
            args.train_source = None

    # Resolve relative paths
    args.train_data = str(Path(args.train_data).resolve())
    args.b8_checkpoint = str(Path(args.b8_checkpoint).resolve())
    if args.dev_data:
        args.dev_data = str(Path(args.dev_data).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())

    return args


def setup_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
        stream=sys.stderr,
    )


def main() -> None:
    args = parse_args()
    setup_logging()
    logger.info("Arguments: %s", vars(args))

    if not torch.cuda.is_available():
        logger.error("CUDA is not available.  Training requires a GPU.")
        sys.exit(1)

    try:
        exit_code = train_bc8(args)
    except Exception:
        logger.exception("Training failed unexpectedly")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
