#!/usr/bin/env python3
"""Evaluate the BC8 QLoRA model on eval50.json.

Loads the base model with 4-bit quantization (eval-only, no kbit training
prep which casts to fp32 and blows up memory), applies the BC8-v3 LoRA
adapter, and runs generation on dev samples using plain-text prompts.
Saves results to checkpoints/BC8-v3/eval_result.json.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cli.train_b8 import (
    load_dev_data,
    render_prompt_text,
    classify_json_errors,
)
from src.eval import parse_json_object
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def load_model_for_eval(model_name: str):
    """Load model in 4-bit NF4 for inference (no kbit training prep)."""
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
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = True
    return model, tokenizer


def main():
    # Unbuffered output so remote_run.py can stream progress
    sys.stdout.reconfigure(line_buffering=True)

    # ---- Config ----
    base_model_name = "Qwen/Qwen3-8B"
    project_root = Path(__file__).resolve().parents[2]
    adapter_path = str(project_root / "checkpoints" / "BC8-v3" / "adapter")
    eval_data_path = str(project_root / "data" / "splits" / "eval50.json")
    output_path = str(project_root / "checkpoints" / "BC8-v3" / "eval_result.json")
    max_new_tokens = 1024

    # ---- Load model ----
    print("Loading base model (4-bit quantized) ...", flush=True)
    model, tokenizer = load_model_for_eval(base_model_name)

    print(f"Loading LoRA adapter from {adapter_path} ...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    model.config.use_cache = True

    # ---- Load dev data ----
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
                max_new_tokens=max_new_tokens,
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
