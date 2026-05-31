"""Generate submit.json using a checkpoint on the eval data."""
import argparse, json, logging, sys, time
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval import parse_json_object
from src.cli.train_b8 import render_prompt_text

logger = logging.getLogger(__name__)


def load_model(base_model: str, device: str, checkpoint: str | None = None):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if checkpoint is None:
        # Prompt-only mode (e.g. P14 baseline): load directly with 4-bit NF4
        model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb,
            device_map={"": device}, torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
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
    parser.add_argument("--checkpoint", default=None, help="LoRA adapter path (omit for prompt-only mode)")
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--no-lora", action="store_true", help="Prompt-only mode (no LoRA adapter)")
    parser.add_argument("--eval-data", default="data/eval_data.json")
    parser.add_argument("--output", default="submit.json")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.no_lora:
        args.checkpoint = None
    elif not args.checkpoint:
        parser.error("--checkpoint is required unless --no-lora is set")

    tasks = json.load(open(args.eval_data))
    n = len(tasks)
    logger.info("Loaded %d tasks", n)

    logger.info("Loading model ...")
    model, tokenizer = load_model(args.base_model, args.device, args.checkpoint)
    logger.info("Model device: %s", model.device)

    submit = []
    report_every = max(1, n // 10)
    t0 = time.monotonic()
    errors = 0

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
        parsed, _parse_errs = parse_json_object(raw)

        valid = True
        if parsed is None:
            valid = False
            logger.warning("idx=%d: parse failed", task["idx"])
        else:
            # Validate required fields
            for field in ("ans_qa_words", "ans_qa_sents"):
                if field not in parsed or not isinstance(parsed[field], dict):
                    valid = False
                    logger.warning("idx=%d: missing or invalid %s", task["idx"], field)
                    break

            if valid and "choose_id" not in parsed:
                valid = False
                logger.warning("idx=%d: missing choose_id", task["idx"])

            if valid:
                choose_opts = task.get("choose", {})
                if choose_opts and parsed["choose_id"] not in choose_opts:
                    valid = False
                    logger.warning("idx=%d: invalid choose_id '%s', options=%s",
                                   task["idx"], parsed["choose_id"], list(choose_opts.keys()))

            # Key coverage warnings (non-blocking: log but don't invalidate)
            if valid and parsed is not None:
                expected_words = set(task.get("qa_words", []))
                got_words = set(parsed.get("ans_qa_words", {}))
                if expected_words - got_words:
                    logger.warning("idx=%d: missing word keys: %s",
                                   task["idx"], sorted(expected_words - got_words))

                expected_sents = set(task.get("qa_sents", []))
                got_sents = set(parsed.get("ans_qa_sents", {}))
                if expected_sents - got_sents:
                    logger.warning("idx=%d: missing sentence keys: %s",
                                   task["idx"], sorted(expected_sents - got_sents))

        if valid:
            submit.append({
                "idx": task["idx"],
                "ans_qa_words": parsed["ans_qa_words"],
                "ans_qa_sents": parsed["ans_qa_sents"],
                "choose_id": parsed["choose_id"],
            })
        else:
            errors += 1
            # Fallback: empty answers
            submit.append({
                "idx": task["idx"],
                "ans_qa_words": {w: "" for w in task.get("qa_words", [])},
                "ans_qa_sents": {s: "" for s in task.get("qa_sents", [])},
                "choose_id": "",
            })

        if (i + 1) % report_every == 0:
            elapsed = time.monotonic() - t0
            logger.info("  %d/%d (%.1fs)  errors=%d", i + 1, n, elapsed, errors)

    total_time = time.monotonic() - t0
    logger.info("Done. Total: %.1fs. Valid: %d/%d (%.1f%%)",
                total_time, n - errors, n, 100 * (n - errors) / n)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(submit, f, ensure_ascii=False, indent=2)
    logger.info("Written %d entries to %s", len(submit), args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    main()
