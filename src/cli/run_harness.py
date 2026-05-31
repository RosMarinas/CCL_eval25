#!/usr/bin/env python3
"""Phase 7: true reasoner-to-formatter harness evaluation for CCL25.

H1: reasoner -> local validator -> local sentiment mapper -> final validator
H2: reasoner -> local validator -> formatter -> final validator

This script follows the contract in docs/contracts/harness.md:
- reasoner emits evidence + sentiment + draft_answer
- draft_answer does not include choose_id
- final choose_id comes from formatter or local mapping
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from peft import PeftModel

from src.cli.train_b8 import load_dev_data, load_quantized_model
from src.cli.train_bc8 import render_evidence_draft_text
from src.eval import (
    CORE_JSON_ERRORS,
    HARD_JSON_ERRORS,
    classify_json_errors as eval_classify_json_errors,
    compute_json_error_rates,
)
from src.harness import (
    build_formatter_input,
    build_formatter_prompt,
    decide_next_action,
    parse_reasoner_output,
    run_harness_once,
    validate_reasoner_output,
)


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]


def _render_retry_prompt(task: dict[str, Any]) -> str:
    return (
        render_evidence_draft_text(task)
        + "\n\n只输出指定 JSON schema。必须包含 idx、evidence、sentiment、draft_answer。"
        + " draft_answer 不包含 choose_id。sentiment.primary 必须使用受控词汇表标签。"
    )


def _generate_with_model(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
    return tokenizer.decode(generated_ids, skip_special_tokens=True), latency


def _load_peft_model(base_model: str, checkpoint_path: str) -> tuple[torch.nn.Module, Any]:
    model, tokenizer = load_quantized_model(base_model)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    if hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = True
    return model, tokenizer


def generate_reasoner_outputs(
    tasks: list[dict[str, Any]],
    checkpoint_path: str,
    base_model: str,
    max_new_tokens: int,
    cache_path: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    model, tokenizer = _load_peft_model(base_model, checkpoint_path)
    outputs = []

    for task in tasks:
        raw_output, latency = _generate_with_model(
            model,
            tokenizer,
            render_evidence_draft_text(task),
            max_new_tokens=max_new_tokens,
        )
        parsed = parse_reasoner_output(raw_output)
        report = validate_reasoner_output(task, parsed)
        retried = False

        if decide_next_action(report) == "retry_reasoner":
            retried = True
            raw_output, latency = _generate_with_model(
                model,
                tokenizer,
                _render_retry_prompt(task),
                max_new_tokens=max_new_tokens,
            )
            parsed = parse_reasoner_output(raw_output)

        outputs.append(
            {
                "idx": task.get("idx"),
                "raw_output": raw_output,
                "parsed": parsed,
                "latency_sec": round(latency, 3),
                "reasoner_retried": retried,
            }
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        for row in outputs:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return outputs


def _make_formatter_fn(base_model: str) -> Callable[[dict[str, Any]], str]:
    model, tokenizer = load_quantized_model(base_model)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    def formatter_fn(formatter_input: dict[str, Any]) -> str:
        prompt = build_formatter_prompt(formatter_input)
        output, _ = _generate_with_model(model, tokenizer, prompt, max_new_tokens=768)
        return output

    return formatter_fn


def run_mode(
    tasks: list[dict[str, Any]],
    reasoner_outputs: list[dict[str, Any]],
    formatter_fn: Callable[[dict[str, Any]], str] | None,
    output_path: Path,
    mode: str,
) -> dict[str, Any]:
    rows = []
    final_errors_all = []
    formatter_called = 0
    fallback_used = 0
    retry_count = 0

    for task, reasoner_row in zip(tasks, reasoner_outputs):
        result = run_harness_once(task, reasoner_row.get("raw_output"), formatter_fn=formatter_fn)
        final_answer = result["final_answer"]
        final_errors = eval_classify_json_errors(
            json.dumps(final_answer, ensure_ascii=False),
            task,
            parsed_json=final_answer,
        )
        rows.append(
            {
                "idx": task.get("idx"),
                "reasoner_output": result["reasoner_output"],
                "validator_report": result["validator_report"],
                "final_validator_report": result["final_validator_report"],
                "final_answer": final_answer,
                "final_error_categories": final_errors,
                "formatter_called": result["formatter_called"],
                "fallback_used": result["fallback_used"],
                "reasoner_retried": reasoner_row.get("reasoner_retried", False),
            }
        )
        final_errors_all.append(final_errors)
        formatter_called += int(result["formatter_called"])
        fallback_used += int(result["fallback_used"])
        retry_count += int(reasoner_row.get("reasoner_retried", False))

    rates = compute_json_error_rates(final_errors_all)
    hard_errors = sum(1 for cats in final_errors_all if set(cats) & HARD_JSON_ERRORS)
    core_errors = sum(1 for cats in final_errors_all if set(cats) & CORE_JSON_ERRORS)
    total = len(rows)
    summary = {
        "mode": mode,
        "sample_count": total,
        "json_error_rate": rates["json_error_rate"],
        "hard_json_error_rate": rates["hard_json_error_rate"],
        "format_style_error_rate": rates["format_style_error_rate"],
        "core_json_error_count": core_errors,
        "hard_json_error_count": hard_errors,
        "formatter_call_rate": formatter_called / total if total else 0.0,
        "fallback_rate": fallback_used / total if total else 0.0,
        "retry_rate": retry_count / total if total else 0.0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the true CCL25 harness.")
    parser.add_argument("--reasoner-checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--dev-data", required=True)
    parser.add_argument("--mode", choices=["h1", "h2"], default="h1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tasks = load_dev_data(args.dev_data)
    output_dir = Path(args.output_dir)
    cache_path = output_dir / f"{args.mode}-reasoner.jsonl"
    detail_path = output_dir / f"{args.mode}-details.jsonl"

    reasoner_outputs = generate_reasoner_outputs(
        tasks,
        checkpoint_path=args.reasoner_checkpoint,
        base_model=args.base_model,
        max_new_tokens=args.max_new_tokens,
        cache_path=cache_path,
        force=args.force,
    )

    formatter_fn = _make_formatter_fn(args.base_model) if args.mode == "h2" else None
    summary = run_mode(tasks, reasoner_outputs, formatter_fn, detail_path, args.mode.upper())
    logger.info(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
