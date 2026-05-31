"""Evaluate a checkpoint on the full eval set.

Usage:
    python tests/full_eval.py --checkpoint checkpoints/B8-v2-lr5e5-ep2/adapter --device cuda:0 --name B8-v2
    python tests/full_eval.py --checkpoint checkpoints/BC8-v2-lr5e5-ep2/adapter --device cuda:1 --name BC8-v2
"""
import argparse, json, logging, os, sys, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval import classify_json_errors, parse_json_object, CORE_JSON_ERRORS, HARD_JSON_ERRORS, FORMAT_STYLE_ERRORS
from src.cli.train_b8 import render_prompt_text

logger = logging.getLogger(__name__)


def load_model(base_model: str, checkpoint: str, device: str):
    """Load 4-bit base model + LoRA on a single GPU."""
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb,
        device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, checkpoint)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--eval-data", default="data/eval_data.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--name", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Ensure we only see the target GPU
    gpu_id = args.device.split(":")[1] if ":" in args.device else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    tasks = json.load(open(args.eval_data))
    n = len(tasks)
    logger.info("[%s] Loaded %d eval samples", args.device, n)

    logger.info("[%s] Loading model ...", args.device)
    model, tokenizer = load_model(args.base_model, args.checkpoint, "cuda:0")
    logger.info("[%s] Model device: %s", args.device, model.device)

    logger.info("[%s] Starting generation on %d samples ...", args.device, n)
    report_every = max(1, n // 10)
    results = []
    t0 = time.monotonic()
    error_idx_list = []

    for i, task in enumerate(tasks):
        prompt = render_prompt_text(task)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        parsed, parse_errs = parse_json_object(raw)
        err_cats = classify_json_errors(raw, task, parsed, parse_errs)
        results.append({"idx": task["idx"], "raw_output": raw, "parsed": parsed,
                        "parse_errors": parse_errs, "error_categories": err_cats})
        if err_cats:
            error_idx_list.append(task["idx"])
        if (i + 1) % report_every == 0:
            elapsed = time.monotonic() - t0
            logger.info("[%s] %d/%d (%.1fs, ~%.1fs/sample)  errors=%d",
                        args.device, i + 1, n, elapsed, elapsed / (i + 1), len(error_idx_list))

    total_time = time.monotonic() - t0
    logger.info("[%s] Done. Total: %.1fs (%.1fs/sample)", args.device, total_time, total_time / n)

    core_count = sum(1 for r in results if set(r["error_categories"]) & CORE_JSON_ERRORS)
    hard_count = sum(1 for r in results if set(r["error_categories"]) & HARD_JSON_ERRORS)
    format_count = sum(1 for r in results if set(r["error_categories"]) & FORMAT_STYLE_ERRORS)

    summary = {
        "experiment": args.name or Path(args.checkpoint).parent.name,
        "checkpoint": args.checkpoint, "sample_count": n,
        "json_error_rate": core_count / n,
        "hard_json_error_rate": hard_count / n,
        "format_style_error_rate": format_count / n,
        "core_error_count": core_count,
        "hard_error_count": hard_count,
        "format_error_count": format_count,
        "total_time_s": total_time,
        "error_indices": sorted(error_idx_list),
    }
    print()
    print("=" * 60)
    print(f"Experiment: {summary['experiment']}")
    print(f"Device:     {args.device}")
    print(f"Samples:    {n}")
    print(f"  json_error_rate:        {summary['json_error_rate']:.4f}  ({core_count}/{n})")
    print(f"  hard_json_error_rate:   {summary['hard_json_error_rate']:.4f}  ({hard_count}/{n})")
    print(f"  format_style_error_rate: {summary['format_style_error_rate']:.4f}  ({format_count}/{n})")
    print(f"  error indices: {error_idx_list}")
    print(f"  total_time: {total_time:.1f}s")
    print("=" * 60)

    output_path = args.output or str(Path(args.checkpoint).parent / "full_eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
    logger.info("[%s] Saved to %s", args.device, output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    main()
