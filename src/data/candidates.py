"""Synthetic perturbed candidate answers for teacher-critique training data.

Takes filtered short-evidence JSONL (correct answers), applies perturbation
types to create wrong candidate answers, outputs JSONL for the
teacher-critique API.

Extracted from src/cli/generate_candidates.py.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

# ============================================================
# Opposite pairs for emotion_flip perturbation
# ============================================================

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

PERTURBATION_TYPES = ["emotion_flip", "word_swap", "sentence_omit"]
PERTURBATION_WEIGHTS = [0.4, 0.3, 0.3]


def build_candidate_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy evidence, sentiment, draft_answer into a candidate dict."""
    return {
        "evidence": copy.deepcopy(record.get("evidence", {})),
        "sentiment": copy.deepcopy(record.get("sentiment", {})),
        "draft_answer": copy.deepcopy(record.get("draft_answer", {})),
    }


def _apply_emotion_flip(candidate: dict[str, Any], compat_mode: bool = False) -> bool:
    sentiment = candidate.get("sentiment", {})
    if not isinstance(sentiment, dict):
        return False
    primary = sentiment.get("primary", "")
    if primary not in EMOTION_OPPOSITE_PAIRS:
        return False
    sentiment["primary"] = EMOTION_OPPOSITE_PAIRS[primary]
    return True


def _apply_word_swap(candidate: dict[str, Any], compat_mode: bool = False) -> bool:
    draft_answer = candidate.get("draft_answer", {})
    if not isinstance(draft_answer, dict):
        return False
    ans_qa_words = draft_answer.get("ans_qa_words", {})
    if not isinstance(ans_qa_words, dict) or len(ans_qa_words) < 1:
        return False

    wrong_meanings = WRONG_MEANINGS_SHORT if compat_mode else WRONG_MEANINGS_LONG
    n_swap = min(random.randint(1, 2), len(ans_qa_words))
    word_keys = random.sample(list(ans_qa_words.keys()), n_swap)

    evidence = candidate.get("evidence", {})
    ev_words = evidence.get("words", {}) if isinstance(evidence, dict) else {}

    for word_key in word_keys:
        wrong = random.choice(wrong_meanings)
        ans_qa_words[word_key] = wrong
        if isinstance(ev_words, dict) and word_key in ev_words:
            ev_entry = ev_words[word_key]
            if isinstance(ev_entry, dict):
                ev_entry["meaning"] = wrong
    return True


def _apply_sentence_omit(candidate: dict[str, Any], compat_mode: bool = False) -> bool:
    draft_answer = candidate.get("draft_answer", {})
    if not isinstance(draft_answer, dict):
        return False
    ans_qa_sents = draft_answer.get("ans_qa_sents", {})
    if not isinstance(ans_qa_sents, dict) or len(ans_qa_sents) < 1:
        return False

    sent_key = random.choice(list(ans_qa_sents.keys()))
    del ans_qa_sents[sent_key]

    evidence = candidate.get("evidence", {})
    if isinstance(evidence, dict):
        ev_sentences = evidence.get("sentences", {})
        if isinstance(ev_sentences, dict) and sent_key in ev_sentences:
            del ev_sentences[sent_key]
    return True


PERTURBATION_FUNCTIONS: dict[str, Any] = {
    "emotion_flip": _apply_emotion_flip,
    "word_swap": _apply_word_swap,
    "sentence_omit": _apply_sentence_omit,
}


def apply_random_perturbation(candidate: dict[str, Any], compat_mode: bool) -> str | None:
    chosen = random.choices(PERTURBATION_TYPES, weights=PERTURBATION_WEIGHTS, k=1)[0]
    remaining = [t for t in PERTURBATION_TYPES if t != chosen]
    random.shuffle(remaining)
    trial_order = [chosen] + remaining
    for ptype in trial_order:
        func = PERTURBATION_FUNCTIONS[ptype]
        if func(candidate, compat_mode):
            return ptype
    return None


def process(input_path: Path, output_path: Path, seed: int,
            compat_mode: bool) -> dict[str, Any]:
    """Read filtered short-evidence JSONL, apply perturbations, write candidates."""
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

            out_line = {"task": record.get("task", {}), "candidate": candidate}
            out_f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
            succeeded += 1
            type_counts[perturbation_type] = type_counts.get(perturbation_type, 0) + 1

    return {"total": total, "succeeded": succeeded, "failed": failed,
            "type_counts": type_counts}
