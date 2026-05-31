#!/usr/bin/env python3
"""P14 Baseline: prompt-only 14B evaluation with batched inference.

Evaluates Qwen3-14B with 4-bit NF4 quantization (no LoRA) on the full
327 eval samples. Uses left-padded batched generation for throughput.

Usage:
    uv run python src/cli/eval_p14.py \
      --base-model Qwen/Qwen3-14B \
      --device cuda:0 \
      --output data/baseline/P14/full_eval.json
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import (
    CORE_JSON_ERRORS,
    FORMAT_STYLE_ERRORS,
    HARD_JSON_ERRORS,
    classify_json_errors,
    parse_json_object,
)
from src.cli.train_b8 import render_prompt_text

logger = logging.getLogger(__name__)


def load_model(base_model: str, device: str):
    """Load BF16 model with 4-bit NF4 quantization (no LoRA).

    Matches the BitsAndBytes approach used by all other eval scripts.
    """
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_batch(model, tokenizer, prompts, max_new_tokens):
    """Batched greedy generation with left-padding.

    Returns list of decoded output strings (one per prompt).
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    batch_max_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(len(prompts)):
        seq = output_ids[i, batch_max_len:]
        results.append(tokenizer.decode(seq, skip_special_tokens=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B")
    parser.add_argument("--eval-data", default="data/eval_data.json")
    parser.add_argument("--output", default="data/baseline/P14/full_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    tasks = json.load(open(args.eval_data))
    n = len(tasks)
    logger.info("Loaded %d eval samples", n)

    logger.info("Loading model %s ...", args.base_model)
    t_load = time.monotonic()
    model, tokenizer = load_model(args.base_model, args.device)
    logger.info("Model loaded in %.1fs on %s", time.monotonic() - t_load, model.device)

    prompts = [render_prompt_text(t) for t in tasks]

    logger.info("Generating (%d samples, batch_size=%d) ...", n, args.batch_size)
    sys.stderr.flush()
    results = []
    t0 = time.monotonic()
    error_idx_list = []

    batch_size = args.batch_size
    log_every = max(1, n // batch_size // 10)

    for i in range(0, n, batch_size):
        batch_prompts = prompts[i : i + batch_size]
        batch_tasks = tasks[i : i + batch_size]
        raw_outputs = generate_batch(model, tokenizer, batch_prompts, args.max_new_tokens)

        for j, task in enumerate(batch_tasks):
            raw = raw_outputs[j]
            parsed, parse_errs = parse_json_object(raw)
            err_cats = classify_json_errors(raw, task, parsed, parse_errs)
            results.append({
                "idx": task["idx"],
                "raw_output": raw,
                "parsed": parsed,
                "parse_errors": parse_errs,
                "error_categories": err_cats,
            })
            if err_cats:
                error_idx_list.append(task["idx"])

        done = min(i + batch_size, n)
        batch_num = i // batch_size + 1
        if batch_num == 1 or batch_num % log_every == 0 or done >= n:
            elapsed = time.monotonic() - t0
            logger.info("  %d/%d (%.1fs, ~%.2fs/sample)  errors=%d",
                        done, n, elapsed, elapsed / done, len(error_idx_list))
            sys.stderr.flush()

    total_time = time.monotonic() - t0
    logger.info("Done. Total: %.1fs (%.2fs/sample)", total_time, total_time / n)

    core_count = sum(1 for r in results if set(r["error_categories"]) & CORE_JSON_ERRORS)
    hard_count = sum(1 for r in results if set(r["error_categories"]) & HARD_JSON_ERRORS)
    format_count = sum(1 for r in results if set(r["error_categories"]) & FORMAT_STYLE_ERRORS)

    summary = {
        "experiment": "P14-baseline",
        "base_model": args.base_model,
        "sample_count": n,
        "json_error_rate": core_count / n,
        "hard_json_error_rate": hard_count / n,
        "format_style_error_rate": format_count / n,
        "core_error_count": core_count,
        "hard_error_count": hard_count,
        "format_error_count": format_count,
        "total_time_s": total_time,
        "batch_size": batch_size,
        "error_indices": sorted(error_idx_list),
    }

    print()
    print("=" * 60)
    print(f"Experiment: P14 Baseline (prompt-only 14B)")
    print(f"Model:      {args.base_model}")
    print(f"Device:     {args.device}")
    print(f"Samples:    {n}")
    print(f"  json_error_rate:        {summary['json_error_rate']:.4f}  ({core_count}/{n})")
    print(f"  hard_json_error_rate:   {summary['hard_json_error_rate']:.4f}  ({hard_count}/{n})")
    print(f"  format_style_error_rate: {summary['format_style_error_rate']:.4f}  ({format_count}/{n})")
    print(f"  error indices: {error_idx_list}")
    print(f"  total_time: {total_time:.1f}s")
    print("=" * 60)

    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    logger.info("Saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
