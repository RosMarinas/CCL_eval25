#!/usr/bin/env python3
"""BC8-final answer-only replay training.

Continues training from the BC8 LoRA checkpoint on answer-only data
to minimise JSON format errors (target: 0%).

Strategy (from docs/plans/training-plan.md Section 5):
  - Initialise from the best BC8 checkpoint (not from base model)
  - Train only on answer-only samples (final JSON output)
  - Lower LR: 2e-5 (0.4x of BC8's 5e-5)
  - Short: 0.3-1 epoch

Usage:
    python src/cli/train_replay.py \
        --base-model Qwen/Qwen3-8B \
        --checkpoint checkpoints/BC8-v3/adapter/ \
        --train-data data/training/b8-answer-only.jsonl \
        --dev-data data/splits/eval50.json \
        --output-dir checkpoints/BC8-final \
        --lr 2e-5 \
        --epochs 0.5 \
        --batch-size 4 --grad-accum 16 \
        --seq-length 2048
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

# Reuse helpers from the B8 training pipeline
from src.cli.train_b8 import (
    PadCollator,
    build_training_pairs,
    evaluate,
    load_answer_only,
    load_dev_data,
    load_quantized_model,
    load_train_source,
    prepare_dataset,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_replay(args: argparse.Namespace) -> int:
    """Execute the BC8-final answer-only replay pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    logger.info("Step 1/6: Loading answer-only data and source index ...")
    answer_records = load_answer_only(args.train_data)
    source_index = load_train_source(args.train_source)
    pairs = build_training_pairs(answer_records, source_index)
    if not pairs:
        logger.error("No training pairs.  Aborting.")
        return 1
    logger.info("  %d pairs ready", len(pairs))

    # ---- Step 2: Load base model with 4-bit quant, then attach BC8 adapter ----
    logger.info("Step 2/6: Loading base model + BC8 LoRA adapter ...")
    model, tokenizer = load_quantized_model(args.base_model)
    model = PeftModel.from_pretrained(model, args.checkpoint, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # ---- Step 3: Tokenise dataset ----
    logger.info("Step 3/6: Tokenising training dataset ...")
    dataset = prepare_dataset(pairs, tokenizer, max_length=args.seq_length)

    # ---- Step 4: Train ----
    logger.info("Step 4/6: Starting training ...")
    total_samples = len(dataset)
    steps_per_epoch = max(1, math.ceil(total_samples / (args.batch_size * args.grad_accum)))
    max_steps = max(1, int(steps_per_epoch * args.epochs))
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

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer),
    )
    try:
        trainer = Trainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        trainer = Trainer(**trainer_kwargs, tokenizer=tokenizer)
    trainer.train()

    # ---- Step 5: Save LoRA adapters ----
    logger.info("Step 5/6: Saving LoRA adapters ...")
    adapter_path = output_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info("  Adapters saved to %s", adapter_path)

    # ---- Step 6: Evaluate on dev set ----
    if args.dev_data:
        logger.info("Step 6/6: Evaluating on dev set ...")
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
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
                "  *** Dev JSON error rate is 0.  Target achieved. ***"
            )

    logger.info("BC8-final replay complete.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC8-final answer-only replay training for CCL25 format correction.",
    )

    # ---- Model ----
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="HuggingFace model ID (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to BC8 LoRA adapter directory",
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
        default="checkpoints/BC8-final",
        help="Output directory (default: checkpoints/BC8-final)",
    )

    # ---- Optimisation ----
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate (default: 2e-5)")
    parser.add_argument("--epochs", type=float, default=1.0, help="Training epochs (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size (default: 4)")
    parser.add_argument(
        "--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=2048, help="Max sequence length (default: 2048)"
    )
    parser.add_argument(
        "--warmup-ratio", type=float, default=0.0, help="Warmup ratio (default: 0.0 — no warmup for replay)"
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
    args.checkpoint = str(Path(args.checkpoint).resolve())
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
        exit_code = train_replay(args)
    except Exception:
        logger.exception("Training failed unexpectedly")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
