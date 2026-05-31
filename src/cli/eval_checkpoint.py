#!/usr/bin/env python3
"""Evaluate a QLoRA checkpoint (B8/BC8/BC8-final) on eval50.json.

Loads the base model with 4-bit quantization, applies a LoRA adapter,
and runs generation on dev samples using plain-text prompts.

Usage:
    uv run python src/cli/eval_checkpoint.py \\
        --checkpoint checkpoints/B8/adapter \\
        --output checkpoints/B8/eval_result.json

This replaces the three nearly-identical eval_b8.py, eval_bc8.py, and
eval_bc8_final.py scripts.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from src.training import (
    load_dev_data,
    render_prompt_text,
    classify_json_errors,
)
from src.eval import parse_json_object
from src.inference import load_model_with_lora


def main():
    parser = argparse.ArgumentParser(description="Evaluate a QLoRA checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B",
                        help="HuggingFace model ID")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--eval-data",
                        default="data/splits/eval50.json",
                        help="Path to eval JSON")
    parser.add_argument("--output", required=True,
                        help="Path to save eval result JSON")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for model loading")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    # Resolve relative paths against project root
    project_root = Path(__file__).resolve().parents[2]

    # ---- Load model ----
    print("Loading base model (4-bit quantized) ...", flush=True)
    checkpoint_path = str(project_root / args.checkpoint
                          if not Path(args.checkpoint).is_absolute()
                          else args.checkpoint)
    print(f"Loading model + LoRA adapter from {checkpoint_path} ...", flush=True)
    model, tokenizer = load_model_with_lora(args.base_model, checkpoint_path,
                                             device=args.device)
    model.eval()
    model.config.use_cache = True

    # ---- Load dev data ----
    eval_data_path = str(project_root / args.eval_data
                         if not Path(args.eval_data).is_absolute()
                         else args.eval_data)
    print(f"Loading eval data from {eval_data_path} ...", flush=True)
    dev_tasks = load_dev_data(eval_data_path)
    print(f"  {len(dev_tasks)} samples loaded", flush=True)

    # ---- Evaluate ----
    core_error_set = {
        "parse_error",
        "missing_top_field",
        "idx_mismatch",
        "wrong_field_type",
        "missing_word_key",
        "missing_sentence_key",
        "empty_required_answer",
        "invalid_choose_id",
    }

    json_errors = 0
    error_details = []

    for i, task in enumerate(dev_tasks):
        prompt = render_prompt_text(task)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed, _ = parse_json_object(raw)
        errors = classify_json_errors(parsed, task)

        has_core_error = bool(set(errors) & core_error_set)
        if has_core_error:
            json_errors += 1
            error_details.append({
                "idx": task["idx"],
                "errors": errors,
                "raw": raw[:200],
            })

        status = "ERR" if has_core_error else "OK"
        print(f"  [{i+1}/{len(dev_tasks)}] idx={task['idx']} -> {status}  errors={errors}", flush=True)

    # ---- Compute stats ----
    total = len(dev_tasks)
    result = {
        "json_error_rate": json_errors / total if total else 0.0,
        "error_count": json_errors,
        "total": total,
        "error_details": error_details,
    }

    # ---- Save ----
    output_path = str(project_root / args.output
                      if not Path(args.output).is_absolute()
                      else args.output)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults saved to {out_path}", flush=True)

    # ---- Report ----
    print(f"\n{'='*50}", flush=True)
    print(f"JSON error rate: {result['json_error_rate']:.4f}", flush=True)
    print(f"Error count: {result['error_count']} / {result['total']}", flush=True)
    print(f"{'='*50}", flush=True)

    if error_details:
        print(f"\nError samples ({len(error_details)} total, showing up to 10):", flush=True)
        for e in error_details[:10]:
            print(f"  idx={e['idx']}: {e['errors']}", flush=True)
            print(f"    raw={e['raw'][:120]}", flush=True)
            print(flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
