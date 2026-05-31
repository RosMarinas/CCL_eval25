"""Re-run failed submission samples with robust JSON extraction.

Usage: python3 remote_run.py python tests/fix_submit_errors.py
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli.train_b8 import render_prompt_text
from src.cli.eval_p14 import load_model, generate_batch

ERROR_INDICES = [213, 256, 287, 311, 317, 319, 320, 321, 322]
SUBMIT_FILE = "submit_p14.json"
BASE_MODEL = "Qwen/Qwen3-14B"
DEVICE = "cuda:0"
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 4

# Find fenced JSON blocks anywhere in text (not just covering the full text)
FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def robust_parse(raw: str):
    """Try multiple strategies to extract valid submission JSON from raw text."""

    # Strategy 1: extract all fenced blocks and try each
    for block in FENCE_RE.findall(raw):
        block = block.strip()
        if not block:
            continue
        result = _try_parse_submit_json(block)
        if result:
            return result

    # Strategy 2: raw_decode at each { position
    decoder = json.JSONDecoder()
    for i, c in enumerate(raw):
        if c != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            result = _check_submit_json(parsed)
            if result:
                return result

    return None


def _try_parse_submit_json(text: str):
    """Try to parse a single JSON string, returning submit data if valid."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try to fix common issues
        return _try_fix_and_parse(text)
    return _check_submit_json(parsed)


def _try_fix_and_parse(text: str):
    """Try to fix common JSON errors and parse."""
    # Fix malformed idx like "idx": 31:1 → should be 311
    fixed = re.sub(r'"idx"\s*:\s*\d+\s*:\s*\d+', lambda m: f'"idx": {m.group(0).split(":")[1].strip()}{m.group(0).split(":")[2].strip().rstrip(",")}', text)
    # Also fix: "idx": 32:0, → {"idx": 320}
    fixed = re.sub(r'"idx"\s*:\s*(\d+):(\d+)', r'"idx": \1\2', fixed)
    try:
        parsed = json.loads(fixed)
    except json.JSONDecodeError:
        return None
    return _check_submit_json(parsed)


def _check_submit_json(parsed: dict):
    """Check if parsed dict has valid submission structure, return normalized dict or None."""
    if not isinstance(parsed, dict):
        return None
    aw = parsed.get("ans_qa_words")
    aq = parsed.get("ans_qa_sents")
    ci = parsed.get("choose_id")
    if not isinstance(aw, dict) or not isinstance(aq, dict) or not isinstance(ci, str):
        return None
    return {"ans_qa_words": aw, "ans_qa_sents": aq, "choose_id": ci}


def main():
    tasks = json.load(open("data/eval_data.json"))
    error_tasks = [t for t in tasks if t["idx"] in ERROR_INDICES]
    print(f"Re-running {len(error_tasks)} failed samples: {ERROR_INDICES}", flush=True)

    print("Loading model ...", flush=True)
    t0 = time.monotonic()
    model, tokenizer = load_model(BASE_MODEL, DEVICE)
    print(f"Model loaded in {time.monotonic()-t0:.1f}s", flush=True)

    prompts = [render_prompt_text(t) for t in error_tasks]

    print("Generating ...", flush=True)
    raw_outputs = generate_batch(model, tokenizer, prompts, MAX_NEW_TOKENS)

    new_entries = {}
    for task, raw in zip(error_tasks, raw_outputs):
        idx = task["idx"]
        result = robust_parse(raw)
        if result:
            entry = {"idx": idx, **result}
            new_entries[idx] = entry
            print(f"  idx={idx}: RECOVERED", flush=True)
            print(f"    choose_id={result['choose_id']!r}", flush=True)
        else:
            print(f"  idx={idx}: STILL INVALID", flush=True)
            print(f"    raw[:500] = {raw[:500]!r}", flush=True)
            print(f"    raw[500:1000] = {raw[500:1000]!r}", flush=True)

    # Merge into submit file
    if new_entries:
        submit = json.load(open(SUBMIT_FILE))
        updated = 0
        for item in submit:
            if item["idx"] in new_entries:
                item.update(new_entries[item["idx"]])
                updated += 1
        json.dump(submit, open(SUBMIT_FILE, "w"), ensure_ascii=False, indent=2)
        print(f"\nMerged {updated}/{len(new_entries)} entries into {SUBMIT_FILE}", flush=True)
    else:
        print("No valid entries to merge.", flush=True)


if __name__ == "__main__":
    main()
