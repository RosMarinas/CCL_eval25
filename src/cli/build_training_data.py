#!/usr/bin/env python3
"""Build training datasets for B8/BC8 QLoRA training.

Three sub-commands:

  1. answer-only    — Assemble B8 answer-only dataset from training data + teacher data.
  2. bc8-mixed      — Assemble BC8 mixed-distillation dataset with proportional sampling.
  3. sentiment-mapping — Assemble sentiment->choose_id mapping data for Formatter training.

Usage:
    python src/cli/build_training_data.py --type answer-only \\
        --keywords data/train-data \\
        --teacher data/teacher/train-short-evidence-filtered.jsonl \\
        --output data/training/b8-answer-only.jsonl

    python src/cli/build_training_data.py --type bc8-mixed \\
        --ratio 50-25-25 \\
        --answer-only data/training/b8-answer-only.jsonl \\
        --short-evidence data/teacher/train-short-evidence-filtered.jsonl \\
        --teacher-critique data/teacher/train-critique-filtered.jsonl \\
        --output-dir data/training/bc8-mixed/

    python src/cli/build_training_data.py --type sentiment-mapping \\
        --teacher data/teacher/dev-short-evidence-filtered.jsonl \\
        --output data/training/sentiment-mapping.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTENCE_DELIMITERS = "。！？!?"

RANDOM_SEED = 42

VALID_TARGETS = frozenset({"final_json", "evidence_draft", "critique_correction"})


# ============================================================
# Path resolution
# ============================================================


def _resolve_path(path_str: str) -> Path:
    """Resolve a path string to an absolute Path.

    If the string is already absolute, use it directly.
    Otherwise, resolve relative to PROJECT_ROOT.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# ============================================================
# Helpers
# ============================================================


def _split_sentences(content: str) -> list[str]:
    """Split content by Chinese sentence-ending punctuation.

    Keeps punctuation attached to each segment. Non-empty segments only.
    Delimiters: 。！？!?
    """
    result: list[str] = []
    current = ""
    for char in content:
        current += char
        if char in SENTENCE_DELIMITERS:
            stripped = current.strip()
            if stripped:
                result.append(stripped)
            current = ""
    remaining = current.strip()
    if remaining:
        result.append(remaining)
    return result


def unique_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate a list preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping unparseable JSONL line: {e}", file=sys.stderr)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as JSONL (one JSON object per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


def resolve_output_dir(path_str: str) -> Path:
    """Resolve output directory path and create it."""
    path = _resolve_path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ============================================================
# Training data loading (preserving keywords object)
# ============================================================


def load_training_samples(keywords_dir: Path) -> list[dict[str, Any]]:
    """Load training samples from a directory tree of train.json files.

    Each original sample has: title, content, keywords (object), trans, emotion.
    We assign sequential idx (0, 1, 2, ...) and preserve the keywords object
    so its keys become qa_words and its values become gold ans_qa_words.

    Returns a list of dicts with keys:
        idx, title, content, keywords, trans, emotion
    """
    samples: list[dict[str, Any]] = []
    idx = 0
    for json_path in sorted(keywords_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Skipping unreadable file {json_path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(
                f"  [WARN] Expected JSON array in {json_path}, got {type(data).__name__}",
                file=sys.stderr,
            )
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            samples.append({
                "idx": idx,
                "title": item.get("title", "") or "",
                "content": item.get("content", "") or "",
                "keywords": item.get("keywords", {}),
                "trans": item.get("trans", "") or "",
                "emotion": item.get("emotion", "") or "",
            })
            idx += 1
    if not samples:
        print(f"  [WARN] No training samples found in directory: {keywords_dir}", file=sys.stderr)
    else:
        print(f"  Loaded {len(samples)} training samples from {keywords_dir}")
    return samples


def index_teacher_data(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Index teacher records by idx for fast lookup.

    Filters out records with parse failures (record_type not short_evidence).
    """
    index: dict[int, dict[str, Any]] = {}
    skipped = 0
    for rec in records:
        if rec.get("record_type") != "short_evidence":
            skipped += 1
            continue
        idx = rec.get("idx")
        if idx is not None:
            index[idx] = rec
    if skipped:
        print(f"  Skipped {skipped} non-short-evidence records in teacher data")
    print(f"  Indexed {len(index)} teacher records by idx")
    return index


# ============================================================
# Validation
# ============================================================


def validate_answer_only(record: dict[str, Any]) -> bool:
    """Validate that an answer-only record has the correct schema.

    Required fields: idx (int), ans_qa_words (dict), ans_qa_sents (dict), choose_id (str).
    """
    if "idx" not in record:
        return False
    if not isinstance(record.get("ans_qa_words"), dict):
        return False
    if not isinstance(record.get("ans_qa_sents"), dict):
        return False
    if not isinstance(record.get("choose_id"), str):
        return False
    # Validate ans_qa_words keys are strings and values are strings
    for key, val in record["ans_qa_words"].items():
        if not isinstance(key, str) or not isinstance(val, str):
            return False
    # Validate ans_qa_sents keys are strings and values are strings
    for key, val in record["ans_qa_sents"].items():
        if not isinstance(key, str) or not isinstance(val, str):
            return False
    return True


def validate_bc8_sample(sample: dict[str, Any]) -> bool:
    """Validate that a BC8 mixed sample has a valid target field."""
    target = sample.get("target")
    if target not in VALID_TARGETS:
        return False
    if target == "final_json":
        return validate_answer_only(sample)
    if target == "evidence_draft":
        # Must have idx, evidence, sentiment, draft_answer
        if "idx" not in sample:
            return False
        if not isinstance(sample.get("evidence"), dict):
            return False
        if not isinstance(sample.get("sentiment"), dict):
            return False
        if not isinstance(sample.get("draft_answer"), dict):
            return False
        return True
    if target == "critique_correction":
        # Must have critique, correction_evidence, corrected_answer
        if not isinstance(sample.get("critique"), dict):
            return False
        if not isinstance(sample.get("correction_evidence"), dict):
            return False
        if not isinstance(sample.get("corrected_answer"), dict):
            return False
        return True
    return False


def validate_sentiment_mapping(record: dict[str, Any]) -> bool:
    """Validate that a sentiment-mapping record has the correct schema."""
    if not isinstance(record.get("sentiment"), str):
        return False
    choose_id = record.get("choose_id", "")
    if choose_id not in ("A", "B", "C", "D", ""):
        print(
            f"  [WARN] Unexpected choose_id={choose_id!r} in sentiment-mapping record",
            file=sys.stderr,
        )
    if not isinstance(record.get("task_choose"), dict):
        return False
    return True


# ============================================================
# Builder: answer-only
# ============================================================


def build_answer_only(args: argparse.Namespace) -> int:
    """Assemble B8 answer-only dataset from training keywords + teacher data.

    1. Load training samples from --keywords directory, preserving keywords dict.
    2. Load teacher data from --teacher JSONL, index by idx.
    3. For each training sample:
       - ans_qa_words = keywords dict directly (gold labels).
       - ans_qa_sents = teacher draft_answer.ans_qa_sents if available,
                        otherwise empty strings keyed by unique qa_sents.
       - choose_id = "" (training data has no options).
    4. Validate each record before writing.
    5. Output: JSONL with one final JSON object per line.
    """
    keywords_dir = _resolve_path(args.keywords)
    teacher_path = _resolve_path(args.teacher)
    output_path = _resolve_path(args.output)

    print(f"[answer-only] Loading training samples from: {keywords_dir}")
    samples = load_training_samples(keywords_dir)
    if not samples:
        print("[ERROR] No training samples loaded; nothing to build.", file=sys.stderr)
        return 1

    print(f"[answer-only] Loading teacher data from: {teacher_path}")
    if not teacher_path.exists():
        print(f"[ERROR] Teacher data not found: {teacher_path}", file=sys.stderr)
        return 1
    teacher_records = load_jsonl(teacher_path)
    teacher_index = index_teacher_data(teacher_records)

    records: list[dict[str, Any]] = []
    validation_errors = 0

    for sample in samples:
        idx = sample["idx"]
        keywords = sample.get("keywords", {})

        # ans_qa_words: gold labels from training data keywords
        ans_qa_words = dict(keywords)

        # ans_qa_sents: from teacher data if available, else empty
        teacher = teacher_index.get(idx, {})
        draft_answer = teacher.get("draft_answer", {})
        if isinstance(draft_answer, dict) and draft_answer.get("ans_qa_sents"):
            ans_qa_sents = dict(draft_answer["ans_qa_sents"])
        else:
            # Fallback: use qa_sents from teacher's task, or split content
            task = teacher.get("task", {})
            qa_sents = task.get("qa_sents", [])
            if not qa_sents:
                qa_sents = _split_sentences(sample["content"])
            ans_qa_sents = {s: "" for s in unique_preserve_order(qa_sents)}

        record = {
            "idx": idx,
            "ans_qa_words": ans_qa_words,
            "ans_qa_sents": ans_qa_sents,
            "choose_id": "",
        }

        if not validate_answer_only(record):
            print(f"  [ERROR] Validation failed for idx={idx}", file=sys.stderr)
            validation_errors += 1
            continue

        records.append(record)

    if validation_errors:
        print(f"  [WARN] {validation_errors} records failed validation and were skipped", file=sys.stderr)

    print(f"[answer-only] Writing {len(records)} records to: {output_path}")
    write_jsonl(records, output_path)
    print("[answer-only] Done.")
    return 0


# ============================================================
# Builder: bc8-mixed
# ============================================================


def parse_ratio(ratio_str: str) -> tuple[int, int, int]:
    """Parse a ratio string like '60-30-10' into (a, b, c)."""
    parts = ratio_str.split("-")
    if len(parts) != 3:
        print(f"[ERROR] Invalid ratio format: {ratio_str!r}. Expected format: '60-30-10'", file=sys.stderr)
        sys.exit(1)
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        print(f"[ERROR] Invalid ratio values: {ratio_str!r}", file=sys.stderr)
        sys.exit(1)


def build_bc8_mixed(args: argparse.Namespace) -> int:
    """Assemble BC8 mixed-distillation dataset.

    1. Read answer-only, short-evidence, and teacher-critique data.
    2. Sample proportionally based on --ratio (default 50-25-25).
    3. Each sample gets a `target` field.
    4. If teacher-critique file doesn't exist, adjust to 70/30 and warn.
    5. Shuffle all samples, then split 90/10 into train/val.
    6. Write to train.jsonl and val.jsonl in --output-dir.
    """
    answer_only_path = _resolve_path(args.answer_only)
    short_evidence_path = _resolve_path(args.short_evidence)
    teacher_critique_path = _resolve_path(args.teacher_critique) if args.teacher_critique else None
    ratio_str = args.ratio
    output_dir = resolve_output_dir(args.output_dir)
    seed = args.seed

    random.seed(seed)

    # Parse ratio
    a_ratio, b_ratio, c_ratio = parse_ratio(ratio_str)

    # Load data sources
    print(f"[bc8-mixed] Loading answer-only data from: {answer_only_path}")
    if not answer_only_path.exists():
        print(f"[ERROR] Answer-only data not found: {answer_only_path}", file=sys.stderr)
        return 1
    answer_only_data = load_jsonl(answer_only_path)
    print(f"  Loaded {len(answer_only_data)} answer-only records")

    print(f"[bc8-mixed] Loading short-evidence data from: {short_evidence_path}")
    if not short_evidence_path.exists():
        print(f"[ERROR] Short-evidence data not found: {short_evidence_path}", file=sys.stderr)
        return 1
    short_evidence_data = load_jsonl(short_evidence_path)

    # Filter short_evidence to valid records only (skip parse failures)
    short_evidence_valid = [
        r for r in short_evidence_data
        if r.get("record_type") == "short_evidence"
    ]
    skipped = len(short_evidence_data) - len(short_evidence_valid)
    if skipped:
        print(f"  Skipped {skipped} non-short-evidence records")
    print(f"  Loaded {len(short_evidence_valid)} valid short-evidence records")

    # Handle teacher-critique (may not exist in first round)
    teacher_critique_data: list[dict[str, Any]] = []
    if teacher_critique_path and teacher_critique_path.exists():
        teacher_critique_data = load_jsonl(teacher_critique_path)
        print(f"  Loaded {len(teacher_critique_data)} teacher-critique records")
    else:
        if teacher_critique_path:
            print(f"  [WARN] Teacher-critique file not found: {teacher_critique_path}", file=sys.stderr)
        print("  [WARN] Teacher-critique not available. Adjusting ratio from", file=sys.stderr)
        print(f"         {a_ratio}-{b_ratio}-{c_ratio} to accommodate.", file=sys.stderr)
        # Redistribute: if original had 50-25-25, become 67-33-0
        if a_ratio > 0 and b_ratio > 0:
            total_ab = a_ratio + b_ratio
            a_ratio = round(a_ratio / total_ab * 100)
            b_ratio = 100 - a_ratio
            c_ratio = 0
            print(f"  [WARN] Adjusted ratio to {a_ratio}-{b_ratio}-{c_ratio}", file=sys.stderr)

    # If teacher-critique exists but is empty or we still have c_ratio > 0 with no data
    if c_ratio > 0 and not teacher_critique_data:
        print(
            "  [WARN] Teacher-critique ratio > 0 but no data available. "
            "Redistributing ratio.",
            file=sys.stderr,
        )
        total_ab = a_ratio + b_ratio
        a_ratio = round(a_ratio / total_ab * 100)
        b_ratio = 100 - a_ratio
        c_ratio = 0
        print(f"  [WARN] Adjusted ratio to {a_ratio}-{b_ratio}-{c_ratio}", file=sys.stderr)

    # Determine target counts maintaining proportions
    # We compute a target total such that each source's available samples
    # are either fully used or sampled down to the required proportion.
    avail_a = len(answer_only_data)
    avail_b = len(short_evidence_valid)
    avail_c = len(teacher_critique_data)

    if avail_a == 0:
        print("[ERROR] No answer-only data available.", file=sys.stderr)
        return 1
    if avail_b == 0:
        print("[ERROR] No short-evidence data available.", file=sys.stderr)
        return 1

    # Determine the maximum samples we can take while respecting proportions
    # For each source i with ratio r_i, if we take n_i samples, n_i / r_i should be equal
    # to the total proportion unit. We want to maximize usage.
    # The constraint is n_i <= avail_i and n_i / r_i = constant.
    # So constant = min(avail_a / r_a, avail_b / r_b, avail_c / r_c) when c is available.
    candidates = [avail_a / a_ratio, avail_b / b_ratio]
    if c_ratio > 0 and avail_c > 0:
        candidates.append(avail_c / c_ratio)
    proportion_unit = min(candidates)

    n_a = int(proportion_unit * a_ratio)
    n_b = int(proportion_unit * b_ratio)
    n_c = int(proportion_unit * c_ratio) if c_ratio > 0 and avail_c > 0 else 0

    # Ensure at least some samples from each available source
    if n_a <= 0 and avail_a > 0:
        n_a = min(1, avail_a)
    if n_b <= 0 and avail_b > 0:
        n_b = min(1, avail_b)

    # Shuffle each source and sample
    random.shuffle(answer_only_data)
    random.shuffle(short_evidence_valid)
    random.shuffle(teacher_critique_data)

    sampled_a = answer_only_data[:n_a]
    sampled_b = short_evidence_valid[:n_b]
    sampled_c = teacher_critique_data[:n_c]

    # Add target field
    for rec in sampled_a:
        rec["target"] = "final_json"
    for rec in sampled_b:
        rec["target"] = "evidence_draft"
    for rec in sampled_c:
        rec["target"] = "critique_correction"

    all_samples: list[dict[str, Any]] = sampled_a + sampled_b + sampled_c

    # Validate
    validated: list[dict[str, Any]] = []
    for sample in all_samples:
        if validate_bc8_sample(sample):
            validated.append(sample)
        else:
            print(f"  [WARN] Validation failed for idx={sample.get('idx', '?')}, target={sample.get('target', '?')}", file=sys.stderr)

    if not validated:
        print("[ERROR] No valid samples after validation.", file=sys.stderr)
        return 1

    # Shuffle and split 90/10
    random.shuffle(validated)
    split_idx = int(len(validated) * 0.9)
    train_data = validated[:split_idx]
    val_data = validated[split_idx:]

    print(f"\n[bc8-mixed] Dataset composition:")
    print(f"  Answer-only:     {len(sampled_a)} (target=final_json)")
    print(f"  Short-evidence:  {len(sampled_b)} (target=evidence_draft)")
    print(f"  Teacher-critique:{len(sampled_c)} (target=critique_correction)")
    print(f"  Total:           {len(all_samples)}")
    print(f"  Validation OK:   {len(validated)}")
    print(f"  Train:           {len(train_data)}")
    print(f"  Val:             {len(val_data)}")

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(train_data, train_path)
    write_jsonl(val_data, val_path)

    # Write a metadata file with stats
    metadata = {
        "ratio": f"{a_ratio}-{b_ratio}-{c_ratio}",
        "answer_only_count": len(sampled_a),
        "short_evidence_count": len(sampled_b),
        "teacher_critique_count": len(sampled_c),
        "validation_ok": len(validated),
        "train_count": len(train_data),
        "val_count": len(val_data),
        "seed": seed,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Wrote metadata to {meta_path}")

    print("[bc8-mixed] Done.")
    return 0


# ============================================================
# Builder: sentiment-mapping
# ============================================================


def build_sentiment_mapping(args: argparse.Namespace) -> int:
    """Assemble sentiment->choose_id mapping data for Formatter training.

    1. Read dev teacher data (which has final_answer.choose_id).
    2. Extract sentiment.primary + choose_id pairs.
    3. Also include task.choose options for context.
    4. Output: JSONL with {sentiment, choose_id, task_choose}.
    """
    teacher_path = _resolve_path(args.teacher)
    output_path = _resolve_path(args.output)

    print(f"[sentiment-mapping] Loading teacher data from: {teacher_path}")
    if not teacher_path.exists():
        print(f"[ERROR] Teacher data not found: {teacher_path}", file=sys.stderr)
        return 1
    records = load_jsonl(teacher_path)

    mapping: list[dict[str, Any]] = []
    skipped_no_final = 0
    skipped_no_sentiment = 0
    skipped_no_choose = 0

    for rec in records:
        # Only process short_evidence records
        if rec.get("record_type") != "short_evidence":
            continue

        # Must have final_answer with choose_id
        final_answer = rec.get("final_answer")
        if not isinstance(final_answer, dict):
            skipped_no_final += 1
            continue

        choose_id = final_answer.get("choose_id", "")
        if not choose_id:
            skipped_no_choose += 1
            continue

        # Must have sentiment
        sentiment = rec.get("sentiment", {})
        if not isinstance(sentiment, dict):
            skipped_no_sentiment += 1
            continue

        primary = sentiment.get("primary", "")
        if not primary:
            skipped_no_sentiment += 1
            continue

        # Must have task.choose
        task = rec.get("task", {})
        task_choose = task.get("choose", {})
        if not isinstance(task_choose, dict) or not task_choose:
            skipped_no_choose += 1
            continue

        # Validate choose_id is in task_choose
        if choose_id not in task_choose:
            print(
                f"  [WARN] idx={rec.get('idx', '?')}: choose_id={choose_id!r} not in "
                f"task.choose keys {set(task_choose)}",
                file=sys.stderr,
            )

        mapping_record = {
            "sentiment": primary,
            "choose_id": choose_id,
            "task_choose": task_choose,
        }

        if not validate_sentiment_mapping(mapping_record):
            print(
                f"  [WARN] Validation failed for idx={rec.get('idx', '?')}",
                file=sys.stderr,
            )
            continue

        mapping.append(mapping_record)

    print(f"\n[sentiment-mapping] Stats:")
    print(f"  Total records read:     {len(records)}")
    print(f"  Skipped (no final_answer): {skipped_no_final}")
    print(f"  Skipped (no sentiment):    {skipped_no_sentiment}")
    print(f"  Skipped (no choose_id):    {skipped_no_choose}")
    print(f"  Mapping records written:   {len(mapping)}")

    if not mapping:
        print("[WARN] No valid sentiment-mapping records found.", file=sys.stderr)
        return 1

    write_jsonl(mapping, output_path)
    print("[sentiment-mapping] Done.")
    return 0


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sub-commands."""
    parser = argparse.ArgumentParser(
        description="Build training datasets for B8/BC8 QLoRA training.",
    )
    subparsers = parser.add_subparsers(dest="type", required=True)

    # ---- answer-only ----
    ao = subparsers.add_parser(
        "answer-only",
        aliases=["answer-only"],
        help="Assemble B8 answer-only dataset from training keywords + teacher data.",
    )
    ao.add_argument(
        "--keywords",
        type=str,
        required=True,
        help="Path to train-data/ directory containing keyword JSON files",
    )
    ao.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Path to teacher data JSONL (e.g. train-short-evidence-filtered.jsonl)",
    )
    ao.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output JSONL file path (e.g. data/training/b8-answer-only.jsonl)",
    )

    # ---- bc8-mixed ----
    bc8 = subparsers.add_parser(
        "bc8-mixed",
        aliases=["bc8-mixed"],
        help="Assemble BC8 mixed-distillation dataset with proportional sampling.",
    )
    bc8.add_argument(
        "--ratio",
        type=str,
        default="50-25-25",
        help="Data mix ratio as A-B-C (default: 50-25-25)",
    )
    bc8.add_argument(
        "--answer-only",
        type=str,
        required=True,
        help="Path to B8 answer-only JSONL data",
    )
    bc8.add_argument(
        "--short-evidence",
        type=str,
        required=True,
        help="Path to short-evidence teacher data JSONL",
    )
    bc8.add_argument(
        "--teacher-critique",
        type=str,
        default=None,
        help="Path to teacher-critique data JSONL (optional, may not exist in first round)",
    )
    bc8.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for train.jsonl and val.jsonl (e.g. data/training/bc8-mixed/)",
    )
    bc8.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for shuffling (default: {RANDOM_SEED})",
    )

    # ---- sentiment-mapping ----
    sm = subparsers.add_parser(
        "sentiment-mapping",
        aliases=["sentiment-mapping"],
        help="Assemble sentiment->choose_id mapping data for Formatter training.",
    )
    sm.add_argument(
        "--teacher",
        type=str,
        required=True,
        help="Path to dev teacher data JSONL (with final_answer.choose_id)",
    )
    sm.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output JSONL file path (e.g. data/training/sentiment-mapping.jsonl)",
    )

    args = parser.parse_args()

    # Normalize sub-command name (strip underscores to match internal function names)
    if args.type in ("answer-only",):
        pass
    elif args.type in ("bc8-mixed",):
        pass
    elif args.type in ("sentiment-mapping",):
        pass

    return args


# ============================================================
# Main entry point
# ============================================================


def main() -> int:
    """Dispatch to the appropriate builder based on --type."""
    args = parse_args()

    print(f"CCL25 Training Data Builder (type={args.type})")
    print(f"{'=' * 50}")

    if args.type == "answer-only":
        return build_answer_only(args)
    elif args.type == "bc8-mixed":
        return build_bc8_mixed(args)
    elif args.type == "sentiment-mapping":
        return build_sentiment_mapping(args)
    else:
        print(f"[FATAL] Unknown type: {args.type}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
