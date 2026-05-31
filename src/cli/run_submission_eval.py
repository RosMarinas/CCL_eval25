#!/usr/bin/env python3
"""Phase 7: direct-final submission evaluation for CCL25.

This script currently evaluates the direct-final/submission path: BC8-v1 is
prompted to emit final JSON directly, then H1/H2 measure rule fallback and
formatter behavior around that final JSON.

It is not yet the true reasoner-to-formatter harness described in docs/contracts/harness.md.
The true harness requires a reasoner prompt that emits evidence + sentiment +
draft_answer without choose_id, then maps sentiment to choose_id in formatter or
local mapper.

H1: BC8-v1 direct-final output + rule-only postprocess (fallback_final)
H2: BC8-v1 direct-final output + Qwen3-8B formatter + final answer

Usage:
    python src/cli/run_submission_eval.py \
        --reasoner-checkpoint checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/ \
        --dev-data data/splits/eval50.json \
        --mode h1 \
        --output-dir data/harness/

    python src/cli/run_submission_eval.py \
        --reasoner-checkpoint checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/ \
        --base-model Qwen/Qwen3-8B \
        --dev-data data/splits/eval50.json \
        --mode h2 \
        --output-dir data/harness/

Strategy:
    1. Generate reasoner outputs for all 50 dev samples (cached to JSONL)
    2. H1: apply rule-only postprocess (fallback_final), measure error rates
    3. H2: load base Qwen3-8B as formatter, run formatter pipeline, compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from src.training import load_dev_data, render_prompt_text
from src.inference import load_model
from src.eval import (
    CORE_JSON_ERRORS,
    HARD_JSON_ERRORS,
    classify_json_errors as eval_classify_json_errors,
    compute_json_error_rates,
    parse_json_object,
)
from src.harness import (
    build_formatter_input,
    build_formatter_prompt,
    fallback_final,
    parse_reasoner_output,
    validate_final_output,
    validate_reasoner_output,
    should_skip_formatter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoner generation
# ---------------------------------------------------------------------------


def generate_reasoner_outputs(
    tasks: list[dict[str, Any]],
    checkpoint_path: str,
    base_model: str,
    max_new_tokens: int,
    cache_path: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Generate reasoner outputs for all tasks using BC8-v1, with caching."""
    if cache_path.exists() and not force:
        logger.info("Loading cached reasoner outputs from %s", cache_path)
        return _load_cached(cache_path)

    logger.info("Generating reasoner outputs for %d tasks ...", len(tasks))

    logger.info("Loading base model: %s", base_model)
    model, tokenizer = load_model(base_model)

    logger.info("Loading LoRA adapter from %s", checkpoint_path)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    # Enable KV cache for fast generation — must be set on base model too
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    # Ensure the underlying base model's use_cache is also True
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = True

    model_device = model.device
    logger.info("Model device: %s", model_device)
    sys.stderr.flush()

    # Warm-up: run one short generation to trigger CUDA kernel compilation
    logger.info("Running warm-up generation (first sample) ...")
    sys.stderr.flush()
    warmup_task = tasks[0]
    warmup_prompt = render_prompt_text(warmup_task)
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model_device)
    with torch.no_grad():
        _ = model.generate(
            **warmup_inputs,
            max_new_tokens=4,  # just a few tokens for warm-up
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    logger.info("Warm-up complete. Starting generation loop ...")
    sys.stderr.flush()

    outputs: list[dict[str, Any]] = []
    num_samples = len(tasks)
    report_interval = max(1, num_samples // 5)

    for i, task in enumerate(tasks):
        if (i + 1) % report_interval == 0:
            logger.info("  Generating %d/%d ...", i + 1, num_samples)
            sys.stderr.flush()

        prompt = render_prompt_text(task)
        inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
        input_len = inputs["input_ids"].shape[1]

        start = time.time()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        latency = time.time() - start

        generated_ids = generated[0][input_len:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, parse_errors = parse_json_object(raw)

        outputs.append({
            "idx": task.get("idx"),
            "raw_output": raw,
            "parsed": parsed,
            "parse_errors": parse_errors,
            "latency_sec": round(latency, 3),
        })

        # Log per-sample latency (short debugging line)
        logger.info("    sample %d/%d done in %.1fs", i + 1, num_samples, latency)
        sys.stderr.flush()

    # Cache to file
    with open(cache_path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    logger.info("Cached %d reasoner outputs to %s", len(outputs), cache_path)

    return outputs


def _load_cached(cache_path: Path) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outputs.append(json.loads(line))
    logger.info("Loaded %d cached reasoner outputs", len(outputs))
    return outputs


# ---------------------------------------------------------------------------
# H1: Rule-only postprocess
# ---------------------------------------------------------------------------


def run_h1(
    reasoner_outputs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """H1: BC8-v1 output -> fallback_final -> validation.

    Treats the BC8-v1 direct output (which is in final JSON format:
    idx + ans_qa_words + ans_qa_sents + choose_id) as a draft_answer
    inside a synthetic reasoner_output, then applies the rule-only
    fallback_final() to produce the final answer.
    """
    logger.info("=" * 50)
    logger.info("H1: Rule-only postprocess")
    logger.info("=" * 50)

    results: list[dict[str, Any]] = []
    all_error_categories: list[list[str]] = []

    for task, ro in zip(tasks, reasoner_outputs):
        parsed = ro.get("parsed")
        raw = ro.get("raw_output", "")
        parse_errs = ro.get("parse_errors", [])

        # Wrap BC8 output as reasoner_output with draft_answer
        wrapped: dict[str, Any] | None = (
            {"draft_answer": parsed} if parsed else None
        )

        # Apply rule-only postprocess
        final_answer = fallback_final(task, wrapped)

        # Classify JSON errors on final answer (pre-computed parse_errors
        # avoids re-parsing raw text)
        error_categories = eval_classify_json_errors(
            raw,
            task,
            parsed_json=final_answer,
            parse_errors=list(parse_errs),
        )

        # Track key coverage
        word_coverage, sent_coverage = _compute_coverage(task, final_answer)

        results.append({
            "idx": task.get("idx"),
            "final_answer": final_answer,
            "error_categories": error_categories,
            "word_coverage": word_coverage,
            "sent_coverage": sent_coverage,
        })
        all_error_categories.append(error_categories)

    # Compute summary metrics
    error_rates = compute_json_error_rates(all_error_categories)
    word_coverage_avg = _avg_coverage(results, "word_coverage")
    sent_coverage_avg = _avg_coverage(results, "sent_coverage")
    core_json_errors = sum(
        1 for cats in all_error_categories if set(cats) & CORE_JSON_ERRORS
    )
    hard_json_errors = sum(
        1 for cats in all_error_categories if set(cats) & HARD_JSON_ERRORS
    )

    summary: dict[str, Any] = {
        "mode": "H1",
        "sample_count": len(results),
        "json_error_rate": error_rates["json_error_rate"],
        "hard_json_error_rate": error_rates["hard_json_error_rate"],
        "format_style_error_rate": error_rates["format_style_error_rate"],
        "core_json_error_count": core_json_errors,
        "hard_json_error_count": hard_json_errors,
        "avg_word_coverage": word_coverage_avg,
        "avg_sent_coverage": sent_coverage_avg,
    }

    # Write per-sample details
    h1_detail_path = output_dir / "H1-details.jsonl"
    with open(h1_detail_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Written H1 details: %s", h1_detail_path)

    # Write summary
    h1_summary_path = output_dir / "H1-summary.json"
    with open(h1_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Written H1 summary: %s", h1_summary_path)

    _log_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# H2: Reasoner + Formatter
# ---------------------------------------------------------------------------


def run_h2(
    reasoner_outputs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """H2: BC8-v1 output -> formatter -> validation.

    Loads a separate Qwen3-8B base model as the formatter and runs each
    sample through the two-stage pipeline. Compares draft (BC8-v1 direct
    output) vs final (formatter output) error rates.
    """
    logger.info("=" * 50)
    logger.info("H2: Reasoner + Formatter")
    logger.info("=" * 50)

    # Load formatter model (base Qwen3-8B, no LoRA)
    logger.info("Loading formatter model: %s", args.base_model)
    formatter_model, formatter_tokenizer = load_model(args.base_model)
    formatter_model.eval()
    formatter_model.config.use_cache = True
    if hasattr(formatter_model, "gradient_checkpointing_disable"):
        formatter_model.gradient_checkpointing_disable()
    formatter_device = formatter_model.device
    logger.info("Formatter model on device: %s", formatter_device)

    # Define formatter generation function
    def formatter_generate(formatter_input: dict[str, Any]) -> str:
        prompt_text = build_formatter_prompt(formatter_input)
        inputs = formatter_tokenizer(prompt_text, return_tensors="pt").to(
            formatter_device
        )
        with torch.no_grad():
            generated = formatter_model.generate(
                **inputs,
                max_new_tokens=768,
                temperature=0.0,
                do_sample=False,
                pad_token_id=(
                    formatter_tokenizer.pad_token_id
                    or formatter_tokenizer.eos_token_id
                ),
            )
        generated_ids = generated[0][inputs["input_ids"].shape[1]:]
        return formatter_tokenizer.decode(generated_ids, skip_special_tokens=True)

    results: list[dict[str, Any]] = []
    draft_error_categories_all: list[list[str]] = []
    final_error_categories_all: list[list[str]] = []
    num_samples = len(tasks)
    report_interval = max(1, num_samples // 5)

    formatter_called_count = 0
    fallback_used_count = 0
    formatter_fix_count = 0  # draft had error, formatter fixed it
    formatter_regression_count = 0  # draft clean, formatter introduced error

    for i, (task, ro) in enumerate(zip(tasks, reasoner_outputs)):
        if (i + 1) % report_interval == 0:
            logger.info("  Processing %d/%d ...", i + 1, num_samples)

        parsed = ro.get("parsed")
        raw = ro.get("raw_output", "")
        parse_errs = ro.get("parse_errors", [])

        # Wrapped reasoner_output for harness functions
        wrapped: dict[str, Any] | None = (
            {"draft_answer": parsed} if parsed else None
        )

        # Classify draft errors (before formatter; pass pre-computed
        # parse_errors to avoid re-parsing raw text)
        draft_categories = eval_classify_json_errors(
            raw, task, parsed_json=parsed, parse_errors=list(parse_errs)
        )
        draft_had_core_error = bool(set(draft_categories) & CORE_JSON_ERRORS)

        # Build formatter input and call formatter
        formatter_failed = True
        if wrapped is not None:
            report = validate_reasoner_output(task, wrapped)
            formatter_input = build_formatter_input(task, wrapped, report)
            formatter_raw = formatter_generate(formatter_input)
            formatter_parsed, _ = parse_json_object(formatter_raw)
            formatter_called_count += 1

            # If formatter output passes validation, use it
            final_report = validate_final_output(task, formatter_parsed)
            if should_skip_formatter(final_report) and formatter_parsed is not None:
                final_answer = formatter_parsed
                formatter_failed = False
            else:
                fallback_used_count += 1
                final_answer = fallback_final(task, wrapped)
                formatter_failed = True
        else:
            # Parse failed — can't call formatter, fallback directly
            formatter_parsed = None
            fallback_used_count += 1
            final_answer = fallback_final(task, None)

        # Classify final errors
        final_categories = eval_classify_json_errors(
            json.dumps(final_answer, ensure_ascii=False) if final_answer else "",
            task,
            parsed_json=final_answer,
        )
        final_has_core_error = bool(set(final_categories) & CORE_JSON_ERRORS)

        # Track formatter fixes and regressions
        if draft_had_core_error and not final_has_core_error:
            formatter_fix_count += 1
        if not draft_had_core_error and final_has_core_error:
            formatter_regression_count += 1

        # Key coverage
        word_coverage, sent_coverage = _compute_coverage(task, final_answer)

        results.append({
            "idx": task.get("idx"),
            "draft_parsed": parsed,
            "formatter_raw": formatter_raw if wrapped is not None else None,
            "formatter_parsed": formatter_parsed,
            "final_answer": final_answer,
            "draft_error_categories": draft_categories,
            "final_error_categories": final_categories,
            "formatter_called": wrapped is not None,
            "fallback_used": wrapped is None or formatter_failed,
            "word_coverage": word_coverage,
            "sent_coverage": sent_coverage,
        })
        draft_error_categories_all.append(draft_categories)
        final_error_categories_all.append(final_categories)

    # Compute error rates
    draft_error_rates = compute_json_error_rates(draft_error_categories_all)
    final_error_rates = compute_json_error_rates(final_error_categories_all)

    word_coverage_avg = _avg_coverage(results, "word_coverage")
    sent_coverage_avg = _avg_coverage(results, "sent_coverage")

    # Formatter correction / regression rates
    total = len(results)
    formatter_call_rate = formatter_called_count / total if total else 0.0
    fallback_rate = fallback_used_count / total if total else 0.0
    formatter_fix_rate = formatter_fix_count / total if total else 0.0
    formatter_regression_rate = formatter_regression_count / total if total else 0.0
    formatter_net_gain = (formatter_fix_count - formatter_regression_count) / total if total else 0.0

    summary: dict[str, Any] = {
        "mode": "H2",
        "sample_count": total,
        # Draft (BC8-v1 direct) error rates
        "draft_json_error_rate": draft_error_rates["json_error_rate"],
        "draft_hard_json_error_rate": draft_error_rates["hard_json_error_rate"],
        # Final (after formatter) error rates
        "final_json_error_rate": final_error_rates["json_error_rate"],
        "final_hard_json_error_rate": final_error_rates["hard_json_error_rate"],
        # Formatter metrics
        "formatter_call_rate": formatter_call_rate,
        "fallback_rate": fallback_rate,
        "formatter_fix_rate": formatter_fix_rate,
        "formatter_regression_rate": formatter_regression_rate,
        "formatter_net_gain": formatter_net_gain,
        # Coverage
        "avg_word_coverage": word_coverage_avg,
        "avg_sent_coverage": sent_coverage_avg,
    }

    # Write per-sample details
    h2_detail_path = output_dir / "H2-details.jsonl"
    with open(h2_detail_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Written H2 details: %s", h2_detail_path)

    # Write summary
    h2_summary_path = output_dir / "H2-summary.json"
    with open(h2_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Written H2 summary: %s", h2_summary_path)

    _log_summary(summary)
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_coverage(
    task: dict[str, Any], answer: dict[str, Any] | None
) -> tuple[float, float]:
    """Compute word key coverage and sentence key coverage."""
    if answer is None:
        return 0.0, 0.0

    target_words = list(dict.fromkeys(task.get("qa_words") or []))
    target_sents = list(dict.fromkeys(task.get("qa_sents") or []))

    answer_words = answer.get("ans_qa_words") or {}
    answer_sents = answer.get("ans_qa_sents") or {}

    word_keys = {str(k) for k in answer_words}
    sent_keys = {str(k) for k in answer_sents}

    word_cov = (
        sum(1 for w in target_words if str(w) in word_keys) / len(target_words)
        if target_words
        else 1.0
    )
    sent_cov = (
        sum(1 for s in target_sents if str(s) in sent_keys) / len(target_sents)
        if target_sents
        else 1.0
    )

    return word_cov, sent_cov


def _avg_coverage(
    results: list[dict[str, Any]], key: str
) -> float:
    values = [r.get(key, 0.0) for r in results]
    return sum(values) / len(values) if values else 0.0


def _log_summary(summary: dict[str, Any]) -> None:
    """Pretty-print summary to stderr."""
    lines = [
        "",
        "=" * 50,
        f"Mode: {summary.get('mode', '?')}",
        f"Samples: {summary.get('sample_count', 0)}",
    ]
    for k, v in summary.items():
        if k in ("mode", "sample_count"):
            continue
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.4f}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("=" * 50)
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 7: Harness evaluation (H1 rule-only / H2 formatter).",
    )

    parser.add_argument(
        "--reasoner-checkpoint",
        required=True,
        help="Path to BC8 LoRA adapter checkpoint (e.g. "
             "checkpoints/BC8-v1-lr5e5-ep1-10steps/adapter/)",
    )
    parser.add_argument(
        "--dev-data",
        default="data/splits/eval50.json",
        help="Path to dev split JSON (default: data/splits/eval50.json)",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Base model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["h1", "h2", "both"],
        help="H1 = rule-only postprocess, H2 = reasoner + formatter, both = H1 then H2",
    )
    parser.add_argument(
        "--output-dir",
        default="data/harness/",
        help="Output directory (default: data/harness/)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=768,
        help="Max generated tokens (default: 768)",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Regenerate reasoner outputs even if cache exists",
    )

    args = parser.parse_args(argv)

    # Resolve relative paths
    args.dev_data = str(Path(args.dev_data).resolve())
    args.reasoner_checkpoint = str(Path(args.reasoner_checkpoint).resolve())
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
        logger.error("CUDA is not available. Harness requires a GPU.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dev tasks
    tasks = load_dev_data(args.dev_data)
    logger.info("Loaded %d dev samples from %s", len(tasks), args.dev_data)

    # Cache path for reasoner outputs (shared between H1 and H2)
    cache_path = output_dir / "reasoner_outputs.jsonl"

    # Generate (or load cached) reasoner outputs
    reasoner_outputs = generate_reasoner_outputs(
        tasks=tasks,
        checkpoint_path=args.reasoner_checkpoint,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        cache_path=cache_path,
        force=args.force_generate,
    )

    if args.mode in ("h1", "both"):
        run_h1(reasoner_outputs, tasks, output_dir)

    if args.mode in ("h2", "both"):
        # Ensure we have the base model for formatter
        run_h2(reasoner_outputs, tasks, args, output_dir)

    logger.info("Phase 7 harness complete.")


if __name__ == "__main__":
    main()
