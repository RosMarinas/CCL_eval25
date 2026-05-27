#!/usr/bin/env python3
"""B8 answer-only QLoRA training for CCL25 format confirmation.

Fine-tunes Qwen3-8B with QLoRA (4-bit NF4, LoRA rank 16) to output
the CCL25 final JSON schema (idx + ans_qa_words + ans_qa_sents +
choose_id).  Shortened 0.3-epoch run targeting the JSON typo errors
identified in the P8 baseline.

Usage:
    python src/cli/train_b8.py \
        --base-model Qwen/Qwen3-8B \
        --train-data data/training/b8-answer-only.jsonl \
        --dev-data data/splits/eval50.json \
        --output-dir checkpoints/B8/ \
        --lora-r 16 --lora-alpha 32 \
        --lr 1e-4 \
        --epochs 0.3 \
        --batch-size 4 --grad-accum 16 \
        --seq-length 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Prompt template (zero-shot, from docs/contracts/prompt-baseline.md Section 4)
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

# Schema: the only permitted top-level keys in the final JSON
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
# Prompt construction
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


# ---------------------------------------------------------------------------
# Chat-template formatting
# ---------------------------------------------------------------------------


def format_chat_sample(
    task: dict[str, Any],
    output_json: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> str:
    """Format one training sample as a full chat conversation string.

    The prompt is rendered from the task, and the expected output is the
    final JSON.  apply_chat_template adds Qwen3 special tokens and ensures
    consistent tokenization.
    """
    prompt = render_prompt_text(task)
    response = json.dumps(output_json, ensure_ascii=False)
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


def format_user_prompt(
    task: dict[str, Any],
    tokenizer: AutoTokenizer,
) -> str:
    """Format only the user turn with the generation-prompt suffix.

    Used for evaluation (the model generates the assistant part).
    """
    prompt = render_prompt_text(task)
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def prepare_dataset(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dataset:
    """Tokenise and produce input_ids + labels with prompt masking.

    Labels for the prompt part (user turn + system header) are set to -100
    so the language-model loss is computed only on the assistant response.
    """
    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []
    all_attention_masks: list[list[int]] = []
    truncated_count = 0

    for task, output in pairs:
        # Full conversation text
        full_text = format_chat_sample(task, output, tokenizer)
        # Prompt-only text (user + generation prompt suffix)
        prompt_text = format_user_prompt(task, tokenizer)

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


def load_quantized_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model in 4-bit NF4 with double quant and bf16 compute dtype."""
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
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> AutoModelForCausalLM:
    """Wrap the model with LoRA adapters on all attention + MLP linear layers."""
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


def parse_json_output(raw: str) -> dict[str, Any] | None:
    """Extract and parse a JSON object from free-form model output."""
    text = raw.strip()
    if not text:
        return None

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First JSON object
    decoder = json.JSONDecoder()
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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
    target_words = _unique(task.get("qa_words", []))
    target_sents = _unique(task.get("qa_sents", []))

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
    max_new_tokens: int = 512,
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
        # Use plain-text prompt (no chat template wrapping) to match the
        # baseline runner format — the chat template produces different
        # token sequences that inflate JSON error rates artificially.
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

        parsed = parse_json_output(raw)
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
# Orchestration
# ---------------------------------------------------------------------------


def train_b8(args: argparse.Namespace) -> int:
    """Execute the B8 training pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    logger.info("Step 1/7: Loading raw data ...")
    source_index = load_train_source(args.train_source)
    answer_records = load_answer_only(args.train_data)

    # ---- Step 2: Build (task, output) training pairs ----
    logger.info("Step 2/7: Constructing training pairs ...")
    pairs = build_training_pairs(answer_records, source_index)
    if not pairs:
        logger.error("No training pairs could be constructed.  Aborting.")
        return 1
    logger.info("  %d pairs ready", len(pairs))

    # ---- Step 3: Load model + tokenizer with 4-bit quant ----
    logger.info("Step 3/7: Loading QLoRA model ...")
    model, tokenizer = load_quantized_model(args.base_model)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.gradient_checkpointing_enable()

    # ---- Step 4: Tokenise dataset ----
    logger.info("Step 4/7: Tokenising training dataset ...")
    dataset = prepare_dataset(pairs, tokenizer, max_length=args.seq_length)

    # ---- Step 5: Train ----
    logger.info("Step 5/7: Starting training ...")
    total_samples = len(dataset)
    steps_per_epoch = max(1, math.ceil(total_samples / (args.batch_size * args.grad_accum)))
    max_steps = max(1, int(steps_per_epoch * args.epochs))
    max_steps = max(10, max_steps)  # ensure minimum meaningful training so warmup doesn't eat all steps
    logging_steps = max(1, max_steps // 10) if max_steps > 1 else 1
    save_steps = max(1, max_steps)

    logger.info(
        "  samples=%d  per_device_bs=%d  grad_accum=%d  effective_bs=%d  "
        "epochs=%.2f  max_steps=%d",
        total_samples, args.batch_size, args.grad_accum,
        args.batch_size * args.grad_accum, args.epochs, max_steps,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=max_steps,
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
        gradient_checkpointing=True,
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
    trainer.train()

    # ---- Step 6: Save LoRA adapters (not full model) ----
    logger.info("Step 6/7: Saving LoRA adapters ...")
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("  Adapters saved to %s", adapter_path)

    # ---- Step 7: Evaluate on dev set ----
    if args.dev_data:
        logger.info("Step 7/7: Evaluating on dev set ...")
        model.gradient_checkpointing_disable()
        dev_tasks = load_dev_data(args.dev_data)
        eval_results = evaluate(model, tokenizer, dev_tasks, args.eval_max_new_tokens)

        eval_path = output_dir / "dev_eval.json"
        eval_path.write_text(
            json.dumps(eval_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("  Eval results written to %s", eval_path)

        if eval_results["json_error_rate"] == 0.0:
            logger.info(
                "  *** Dev JSON error rate is 0 — format confirmed.  "
                "Early-stop criteria satisfied. ***"
            )

    logger.info("B8 training complete.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B8 answer-only QLoRA training for CCL25 format confirmation.",
    )

    # ---- Model ----
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID (default: Qwen/Qwen3-8B)",
    )

    # ---- Data ----
    parser.add_argument(
        "--train-data",
        required=True,
        help="Path to b8-answer-only.jsonl",
    )
    parser.add_argument(
        "--train-source",
        default=None,
        help="Path to train-data/ directory (default: <project>/data/train-data)",
    )
    parser.add_argument(
        "--dev-data",
        default=None,
        help="Path to dev split JSON (e.g. data/splits/eval50.json)",
    )

    # ---- Checkpoint output ----
    parser.add_argument(
        "--output-dir",
        default="checkpoints/B8",
        help="Output directory (default: checkpoints/B8)",
    )

    # ---- LoRA hyper-parameters ----
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)"
    )

    # ---- Optimisation ----
    # Conservative LR for a strong, well-trained instruct model —
    # Qwen3-8B is already highly capable; high LR risks catastrophic
    # forgetting of pre-trained instruction-following behavior.
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate (default: 2e-5)")
    parser.add_argument("--epochs", type=float, default=2.0, help="Training epochs (default: 2.0)")
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

    # ---- Evaluation ----
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=512,
        help="Max generated tokens for dev eval (default: 512)",
    )

    args = parser.parse_args(argv)

    # Resolve --train-source default relative to project root
    if args.train_source is None:
        args.train_source = str(PROJECT_ROOT / "data" / "train-data")

    # Resolve relative paths
    args.train_data = str(Path(args.train_data).resolve())
    args.train_source = str(Path(args.train_source).resolve())
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
        exit_code = train_b8(args)
    except Exception:
        logger.exception("Training failed unexpectedly")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
