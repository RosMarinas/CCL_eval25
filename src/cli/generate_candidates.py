#!/usr/bin/env python3
"""Generate synthetic perturbed candidate answers from filtered short-evidence teacher data.

Takes filtered short-evidence JSONL (correct answers), applies one of three perturbation
types to create "wrong" candidate answers, and outputs a JSONL file where each line
has {task: ..., candidate: ...}. The teacher-critique API then critiques these
wrong answers to generate training data for critique learning.

Usage:
    python src/cli/generate_candidates.py \
        --input data/teacher/train-short-evidence-filtered.jsonl \
        --output data/candidates/train-short-evidence-candidates.jsonl \
        --seed 42

Options:
    --input        Input filtered short-evidence JSONL
    --output       Output candidates JSONL
    --seed         Random seed (default: 42)
    --compat-mode  Shorten word_swap wrong meanings for tests
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any


# ============================================================
# Opposite pairs for emotion_flip perturbation
# ============================================================
# Each pair maps both directions so a simple dict lookup works.

EMOTION_OPPOSITE_PAIRS: dict[str, str] = {
    "惜别感伤": "田园之乐",
    "田园之乐": "惜别感伤",
    "送别不舍": "山水闲适",
    "山水闲适": "送别不舍",
    "离别愁绪": "爱情甜蜜",
    "爱情甜蜜": "离别愁绪",
    "思乡怀远": "隐逸情怀",
    "隐逸情怀": "思乡怀远",
    "忧国伤时": "报国壮志",
    "报国壮志": "忧国伤时",
    "仕途失意": "报国壮志",
}

# ============================================================
# Wrong meaning pools for word_swap perturbation
# ============================================================

WRONG_MEANINGS_SHORT: list[str] = [
    "描述了",
    "表达了作者",
    "美好的事物",
    "与诗歌内容无关的解释",
    "这是明显错误的解释",
]

WRONG_MEANINGS_LONG: list[str] = WRONG_MEANINGS_SHORT + [
    "这是一个很长且明显错误的解释用来测试过滤器的长度检测功能",
]

# ============================================================
# Perturbation type configuration
# ============================================================

PERTURBATION_TYPES = ["emotion_flip", "word_swap", "sentence_omit"]
PERTURBATION_WEIGHTS = [0.4, 0.3, 0.3]


# ============================================================
# Candidate builder
# ============================================================


def build_candidate_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy evidence, sentiment, draft_answer into a candidate dict.

    The candidate dict has the same structure as the model-generated portion
    of a short_evidence record, so it can plug directly into the teacher-critique
    prompt as a "wrong answer" to critique.
    """
    return {
        "evidence": copy.deepcopy(record.get("evidence", {})),
        "sentiment": copy.deepcopy(record.get("sentiment", {})),
        "draft_answer": copy.deepcopy(record.get("draft_answer", {})),
    }


# ============================================================
# Perturbation implementations
# ============================================================


def _apply_emotion_flip(candidate: dict[str, Any], compat_mode: bool = False) -> bool:
    """Flip sentiment.primary to its defined opposite.

    If the current primary label has no defined opposite in EMOTION_OPPOSITE_PAIRS,
    returns False (caller should fallback to another perturbation type).

    Keeps evidence and draft_answer unchanged.
    """
    sentiment = candidate.get("sentiment", {})
    if not isinstance(sentiment, dict):
        return False

    primary = sentiment.get("primary", "")
    if primary not in EMOTION_OPPOSITE_PAIRS:
        return False

    sentiment["primary"] = EMOTION_OPPOSITE_PAIRS[primary]
    return True


def _apply_word_swap(candidate: dict[str, Any], compat_mode: bool = False) -> bool:
    """Replace 1-2 word meanings in draft_answer with clearly wrong meanings.

    Also updates the corresponding entry in evidence.words[*].meaning if it exists.

    Returns False if draft_answer.ans_qa_words is empty or not a dict.
    """
    draft_answer = candidate.get("draft_answer", {})
    if not isinstance(draft_answer, dict):
        return False

    ans_qa_words = draft_answer.get("ans_qa_words", {})
    if not isinstance(ans_qa_words, dict) or len(ans_qa_words) < 1:
        return False

    wrong_meanings = WRONG_MEANINGS_SHORT if compat_mode else WRONG_MEANINGS_LONG

    # Pick 1-2 random words to corrupt
    n_swap = min(random.randint(1, 2), len(ans_qa_words))
    word_keys = random.sample(list(ans_qa_words.keys()), n_swap)

    evidence = candidate.get("evidence", {})
    ev_words = evidence.get("words", {}) if isinstance(evidence, dict) else {}

    for word_key in word_keys:
        wrong = random.choice(wrong_meanings)
        ans_qa_words[word_key] = wrong

        # Propagate the wrong meaning into evidence.words if the key exists
        if isinstance(ev_words, dict) and word_key in ev_words:
            ev_entry = ev_words[word_key]
            if isinstance(ev_entry, dict):
                ev_entry["meaning"] = wrong

    return True


def _apply_sentence_omit(candidate: dict[str, Any], compat_mode: bool = False) -> bool:
    """Remove one sentence from draft_answer.ans_qa_sents.

    Also removes the corresponding entry from evidence.sentences if it exists.

    Returns False if draft_answer.ans_qa_sents is empty or not a dict.
    """
    draft_answer = candidate.get("draft_answer", {})
    if not isinstance(draft_answer, dict):
        return False

    ans_qa_sents = draft_answer.get("ans_qa_sents", {})
    if not isinstance(ans_qa_sents, dict) or len(ans_qa_sents) < 1:
        return False

    sent_key = random.choice(list(ans_qa_sents.keys()))
    del ans_qa_sents[sent_key]

    # Remove from evidence.sentences if the key exists
    evidence = candidate.get("evidence", {})
    if isinstance(evidence, dict):
        ev_sentences = evidence.get("sentences", {})
        if isinstance(ev_sentences, dict) and sent_key in ev_sentences:
            del ev_sentences[sent_key]

    return True


# ============================================================
# Perturbation dispatch
# ============================================================

PERTURBATION_FUNCTIONS: dict[str, Any] = {
    "emotion_flip": _apply_emotion_flip,
    "word_swap": _apply_word_swap,
    "sentence_omit": _apply_sentence_omit,
}


def apply_random_perturbation(candidate: dict[str, Any], compat_mode: bool) -> str | None:
    """Apply a randomly-selected perturbation with automatic fallback.

    The primary type is chosen by weighted random sampling from the three types.
    If it cannot be applied (e.g. no defined opposite for the current sentiment label,
    or not enough words/sentences to perturb), the function falls back to the
    remaining types in random order.

    Returns the perturbation type name on success, or None if all types fail.
    """
    chosen = random.choices(PERTURBATION_TYPES, weights=PERTURBATION_WEIGHTS, k=1)[0]
    remaining = [t for t in PERTURBATION_TYPES if t != chosen]
    random.shuffle(remaining)
    trial_order = [chosen] + remaining

    for ptype in trial_order:
        func = PERTURBATION_FUNCTIONS[ptype]
        if func(candidate, compat_mode):
            return ptype
    return None


# ============================================================
# Processing pipeline
# ============================================================


def process(input_path: Path, output_path: Path, seed: int, compat_mode: bool) -> dict[str, Any]:
    """Read filtered short-evidence JSONL, apply perturbations, write candidates JSONL.

    Returns a statistics dict with counts of success, failure, and type distribution.
    """
    random.seed(seed)

    raw_text = input_path.read_text(encoding="utf-8")
    lines = [line for line in raw_text.strip().split("\n") if line.strip()]

    total = len(lines)
    succeeded = 0
    failed = 0
    type_counts: dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                failed += 1
                continue

            if not isinstance(record, dict):
                failed += 1
                continue

            candidate = build_candidate_from_record(record)
            perturbation_type = apply_random_perturbation(candidate, compat_mode)

            if perturbation_type is None:
                failed += 1
                continue

            out_line = {
                "task": record.get("task", {}),
                "candidate": candidate,
            }
            out_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            succeeded += 1
            type_counts[perturbation_type] = type_counts.get(perturbation_type, 0) + 1

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "type_counts": type_counts,
    }


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic perturbed candidate answers from filtered "
            "short-evidence teacher data for teacher-critique input."
        ),
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input filtered short-evidence JSONL file",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output candidates JSONL file",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible perturbations (default: 42)",
    )
    parser.add_argument(
        "--compat-mode",
        action="store_true",
        default=False,
        help="Shorten word_swap wrong meanings for tests",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    stats = process(
        input_path=input_path,
        output_path=output_path,
        seed=args.seed,
        compat_mode=args.compat_mode,
    )

    # Print summary
    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Seed:    {args.seed}")
    print(f"Compat:  {args.compat_mode}")
    print(f"Total:   {stats['total']}")
    print(f"OK:      {stats['succeeded']}")
    print(f"Failed:  {stats['failed']}")
    print(f"\nPerturbation distribution:")
    for ptype in PERTURBATION_TYPES:
        count = stats["type_counts"].get(ptype, 0)
        pct = count / stats["succeeded"] * 100 if stats["succeeded"] > 0 else 0.0
        print(f"  {ptype}: {count} ({pct:.1f}%)")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
