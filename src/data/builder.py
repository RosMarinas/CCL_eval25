"""Training dataset assembly for B8/BC8 QLoRA training.

Three builders:
  1. answer-only    — Assemble B8 answer-only dataset
  2. bc8-mixed      — Assemble BC8 mixed-distillation dataset
  3. sentiment-mapping — Assemble sentiment->choose_id mapping

Extracted from src/cli/build_training_data.py.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.schema import unique_preserve_order

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SENTENCE_DELIMITERS = "。！？!?"
RANDOM_SEED = 42
VALID_TARGETS = frozenset({"final_json", "evidence_draft", "critique_correction"})


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _split_sentences(content: str) -> list[str]:
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping unparseable JSONL line: {e}",
                      file=sys.stderr)
    return records


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records to {path}")


def resolve_output_dir(path_str: str) -> Path:
    path = _resolve_path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_training_samples(keywords_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    idx = 0
    for json_path in sorted(keywords_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [WARN] Skipping unreadable file {json_path}: {e}",
                  file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"  [WARN] Expected JSON array in {json_path}",
                  file=sys.stderr)
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
        print(f"  [WARN] No training samples found in: {keywords_dir}",
              file=sys.stderr)
    else:
        print(f"  Loaded {len(samples)} training samples from {keywords_dir}")
    return samples


def index_teacher_data(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
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


def validate_answer_only(record: dict[str, Any]) -> bool:
    if "idx" not in record:
        return False
    if not isinstance(record.get("ans_qa_words"), dict):
        return False
    if not isinstance(record.get("ans_qa_sents"), dict):
        return False
    if not isinstance(record.get("choose_id"), str):
        return False
    for key, val in record["ans_qa_words"].items():
        if not isinstance(key, str) or not isinstance(val, str):
            return False
    for key, val in record["ans_qa_sents"].items():
        if not isinstance(key, str) or not isinstance(val, str):
            return False
    return True


def validate_bc8_sample(sample: dict[str, Any]) -> bool:
    target = sample.get("target")
    if target not in VALID_TARGETS:
        return False
    if target == "final_json":
        return validate_answer_only(sample)
    if target == "evidence_draft":
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
        if not isinstance(sample.get("critique"), dict):
            return False
        if not isinstance(sample.get("correction_evidence"), dict):
            return False
        if not isinstance(sample.get("corrected_answer"), dict):
            return False
        return True
    return False


def validate_sentiment_mapping(record: dict[str, Any]) -> bool:
    if not isinstance(record.get("sentiment"), str):
        return False
    choose_id = record.get("choose_id", "")
    if choose_id not in ("A", "B", "C", "D", ""):
        print(f"  [WARN] Unexpected choose_id={choose_id!r}",
              file=sys.stderr)
    if not isinstance(record.get("task_choose"), dict):
        return False
    return True


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_answer_only(*, keywords: str, teacher: str, output: str) -> int:
    keywords_dir = _resolve_path(keywords)
    teacher_path = _resolve_path(teacher)
    output_path = _resolve_path(output)

    print(f"[answer-only] Loading training samples from: {keywords_dir}")
    samples = load_training_samples(keywords_dir)
    if not samples:
        print("[ERROR] No training samples loaded.", file=sys.stderr)
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
        keywords_obj = sample.get("keywords", {})
        ans_qa_words = dict(keywords_obj)

        teacher = teacher_index.get(idx, {})
        draft_answer = teacher.get("draft_answer", {})
        if isinstance(draft_answer, dict) and draft_answer.get("ans_qa_sents"):
            ans_qa_sents = dict(draft_answer["ans_qa_sents"])
        else:
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
        print(f"  [WARN] {validation_errors} records failed validation",
              file=sys.stderr)
    print(f"[answer-only] Writing {len(records)} records to: {output_path}")
    write_jsonl(records, output_path)
    print("[answer-only] Done.")
    return 0


def parse_ratio(ratio_str: str) -> tuple[int, int, int]:
    parts = ratio_str.split("-")
    if len(parts) != 3:
        print(f"[ERROR] Invalid ratio: {ratio_str!r}", file=sys.stderr)
        sys.exit(1)
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        print(f"[ERROR] Invalid ratio values: {ratio_str!r}", file=sys.stderr)
        sys.exit(1)


def build_bc8_mixed(
    *,
    answer_only: str,
    short_evidence: str,
    teacher_critique: str | None = None,
    ratio: str = "50-25-25",
    output_dir: str = "",
    seed: int = RANDOM_SEED,
) -> int:
    random.seed(seed)

    a_ratio, b_ratio, c_ratio = parse_ratio(ratio)

    answer_only_path = _resolve_path(answer_only)
    if not answer_only_path.exists():
        print(f"[ERROR] Answer-only data not found: {answer_only_path}",
              file=sys.stderr)
        return 1
    answer_only_data = load_jsonl(answer_only_path)
    print(f"  Loaded {len(answer_only_data)} answer-only records")

    short_evidence_path = _resolve_path(short_evidence)
    if not short_evidence_path.exists():
        print(f"[ERROR] Short-evidence data not found: {short_evidence_path}",
              file=sys.stderr)
        return 1
    short_evidence_data = load_jsonl(short_evidence_path)
    short_evidence_valid = [
        r for r in short_evidence_data
        if r.get("record_type") == "short_evidence"
    ]
    skipped = len(short_evidence_data) - len(short_evidence_valid)
    if skipped:
        print(f"  Skipped {skipped} non-short-evidence records")
    print(f"  Loaded {len(short_evidence_valid)} valid short-evidence records")

    teacher_critique_data: list[dict[str, Any]] = []
    tc_path = _resolve_path(teacher_critique) if teacher_critique else None
    if tc_path and tc_path.exists():
        teacher_critique_data = load_jsonl(tc_path)
        print(f"  Loaded {len(teacher_critique_data)} teacher-critique records")
    elif tc_path:
        print(f"  [WARN] Teacher-critique file not found: {tc_path}",
              file=sys.stderr)
        if a_ratio > 0 and b_ratio > 0:
            total_ab = a_ratio + b_ratio
            a_ratio = round(a_ratio / total_ab * 100)
            b_ratio = 100 - a_ratio
            c_ratio = 0
            print(f"  Adjusted ratio to {a_ratio}-{b_ratio}-{c_ratio}",
                  file=sys.stderr)

    avail_a = len(answer_only_data)
    avail_b = len(short_evidence_valid)
    avail_c = len(teacher_critique_data)

    if avail_a == 0 or avail_b == 0:
        print("[ERROR] Missing required data.", file=sys.stderr)
        return 1

    candidates = [avail_a / a_ratio, avail_b / b_ratio]
    if c_ratio > 0 and avail_c > 0:
        candidates.append(avail_c / c_ratio)
    proportion_unit = min(candidates)

    n_a = max(int(proportion_unit * a_ratio), min(1, avail_a))
    n_b = max(int(proportion_unit * b_ratio), min(1, avail_b))
    n_c = int(proportion_unit * c_ratio) if c_ratio > 0 and avail_c > 0 else 0

    random.shuffle(answer_only_data)
    random.shuffle(short_evidence_valid)
    random.shuffle(teacher_critique_data)

    sampled_a = answer_only_data[:n_a]
    sampled_b = short_evidence_valid[:n_b]
    sampled_c = teacher_critique_data[:n_c]

    for rec in sampled_a:
        rec["target"] = "final_json"
    for rec in sampled_b:
        rec["target"] = "evidence_draft"
    for rec in sampled_c:
        rec["target"] = "critique_correction"

    all_samples = sampled_a + sampled_b + sampled_c
    validated = [s for s in all_samples if validate_bc8_sample(s)]

    if not validated:
        print("[ERROR] No valid samples.", file=sys.stderr)
        return 1

    random.shuffle(validated)
    split_idx = int(len(validated) * 0.9)

    out_dir = resolve_output_dir(output_dir)
    write_jsonl(validated[:split_idx], out_dir / "train.jsonl")
    write_jsonl(validated[split_idx:], out_dir / "val.jsonl")

    metadata = {
        "ratio": f"{a_ratio}-{b_ratio}-{c_ratio}",
        "answer_only_count": len(sampled_a),
        "short_evidence_count": len(sampled_b),
        "teacher_critique_count": len(sampled_c),
        "validation_ok": len(validated),
        "train_count": len(validated[:split_idx]),
        "val_count": len(validated[split_idx:]),
        "seed": seed,
    }
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Wrote metadata to {meta_path}")
    print("[bc8-mixed] Done.")
    return 0


def build_sentiment_mapping(*, teacher: str, output: str) -> int:
    teacher_path = _resolve_path(teacher)
    output_path = _resolve_path(output)

    print(f"[sentiment-mapping] Loading teacher data from: {teacher_path}")
    if not teacher_path.exists():
        print(f"[ERROR] Teacher data not found: {teacher_path}", file=sys.stderr)
        return 1
    records = load_jsonl(teacher_path)

    mapping: list[dict[str, Any]] = []
    skipped = {"no_final": 0, "no_sentiment": 0, "no_choose": 0}

    for rec in records:
        if rec.get("record_type") != "short_evidence":
            continue
        final_answer = rec.get("final_answer")
        if not isinstance(final_answer, dict):
            skipped["no_final"] += 1
            continue
        choose_id = final_answer.get("choose_id", "")
        if not choose_id:
            skipped["no_choose"] += 1
            continue
        sentiment = rec.get("sentiment", {})
        if not isinstance(sentiment, dict):
            skipped["no_sentiment"] += 1
            continue
        primary = sentiment.get("primary", "")
        if not primary:
            skipped["no_sentiment"] += 1
            continue
        task = rec.get("task", {})
        task_choose = task.get("choose", {})
        if not isinstance(task_choose, dict) or not task_choose:
            skipped["no_choose"] += 1
            continue

        mapping_record = {
            "sentiment": primary,
            "choose_id": choose_id,
            "task_choose": task_choose,
        }
        if not validate_sentiment_mapping(mapping_record):
            print(f"  [WARN] Validation failed for idx={rec.get('idx', '?')}",
                  file=sys.stderr)
            continue
        mapping.append(mapping_record)

    print(f"  Mapping records written: {len(mapping)}")
    if not mapping:
        print("[WARN] No valid records", file=sys.stderr)
        return 1
    write_jsonl(mapping, output_path)
    print("[sentiment-mapping] Done.")
    return 0
