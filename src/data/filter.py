#!/usr/bin/env python3
"""
Filter teacher-generated data JSONL before it enters the training dataset.

Reads teacher data JSONL, applies filtering rules defined in docs/contracts/teacher-data.md
Section 6, and outputs filtered JSONL + a statistics report.

Usage:
    python src/cli/filter_teacher_data.py \\
        --input data/teacher/train-short-evidence.jsonl \\
        --output data/teacher/train-short-evidence-filtered.jsonl \\
        --strict

Options:
    --input   Input JSONL file
    --output  Output filtered JSONL file
    --strict  Discard samples that would normally go to human review
    --report  Path to filtering statistics JSON (default: output + ".report.json")
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from src.schema import unique_preserve_order


# ============================================================
# Controlled vocabulary (docs/contracts/data-schema.md Section 3.2)
# ============================================================

CONTROLLED_VOCABULARY = frozenset({
    "惜别感伤", "送别不舍", "离别愁绪",
    "思乡怀远", "羁旅思归", "故园之思",
    "忧国伤时", "报国壮志", "兴亡之叹",
    "山水闲适", "田园之乐", "隐逸情怀",
    "怀古伤今", "历史沧桑", "昔盛今衰",
    "相思闺怨", "爱情甜蜜", "相思之苦",
    "人生无常", "时光易逝", "仕途失意",
    "边塞征战", "将士艰辛", "厌战思归",
    "其他",
})

# CoT field names forbidden at any level
FORBIDDEN_COT_FIELDS = frozenset({
    "cot", "chain_of_thought", "reasoning", "steps", "analysis",
})

VALID_RECORD_TYPES = frozenset({"short_evidence", "teacher_critique"})

# Patterns for detection
MD_CODE_BLOCK_RE = re.compile(r"```")

PROMPT_RESIDUE_INDICATORS = [
    "你是", "请根据", "硬性要求", "输入题目", "请输出",
    "候选答案", "教师模型",
]

APPRECIATION_PATTERNS = [
    re.compile(r"表达了诗人"),
    re.compile(r"表现了诗人"),
    re.compile(r"抒发了诗人"),
    re.compile(r"传达出诗人"),
    re.compile(r"体现了诗人"),
]

# Antonym pairs for semantic-opposite detection (6.4.1)
ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("茂盛", "枯黄"), ("茂盛", "衰败"), ("茂盛", "枯萎"),
    ("繁荣", "衰败"), ("兴盛", "衰亡"), ("繁盛", "凋敝"),
    ("喜悦", "悲伤"), ("喜悦", "哀愁"), ("喜悦", "伤感"),
    ("快乐", "悲伤"), ("快乐", "哀愁"), ("快乐", "凄凉"),
    ("闲适", "悲伤"), ("闲适", "哀愁"), ("闲适", "忧愁"),
    ("乐", "悲"), ("喜", "哀"), ("欢", "愁"),
    ("盛", "衰"), ("生", "死"),
    ("聚", "散"), ("合", "离"),
    ("甜", "苦"), ("暖", "寒"),
]

# Directional sentiment label groups for 6.4.2
POSITIVE_SENTIMENT_LABELS = frozenset({
    "山水闲适", "田园之乐", "隐逸情怀", "爱情甜蜜",
})
NEGATIVE_SENTIMENT_LABELS = frozenset({
    "惜别感伤", "送别不舍", "离别愁绪",
    "思乡怀远", "羁旅思归", "故园之思",
    "忧国伤时", "相思之苦", "仕途失意",
    "将士艰辛", "厌战思归",
    "怀古伤今", "人生无常", "时光易逝",
    "历史沧桑", "昔盛今衰",
})

POSITIVE_KEYWORDS = frozenset({
    "闲适", "快乐", "喜悦", "美好", "热爱", "陶醉",
    "悠然", "恬淡", "宁静", "安逸", "欢快", "愉悦",
})
NEGATIVE_KEYWORDS = frozenset({
    "悲", "哀", "愁", "苦", "恨", "怨", "伤", "痛",
    "忧", "叹", "寂", "寞", "孤", "独", "离", "别",
    "泪", "凄", "凉", "寒", "冷", "凋", "残", "断",
})

# Fields whose length should NOT be checked (input task, candidate answer)
SKIP_FIELD_CHECK_PREFIXES = ("task", "candidate_answer")


# ============================================================
# Helpers
# ============================================================

def count_cjk(text: str) -> int:
    """Count CJK Unified Ideographs characters in text."""
    return sum(1 for c in text if "一" <= c <= "鿿")






def has_markdown_code_blocks(text: str) -> bool:
    """Check if text contains markdown code block markers (```)."""
    return bool(MD_CODE_BLOCK_RE.search(text))


def has_prompt_residue(text: str) -> bool:
    """Check for obvious prompt residue."""
    return any(ind in text for ind in PROMPT_RESIDUE_INDICATORS)


def has_multi_paragraph_long_reasoning(text: str) -> bool:
    """Check for multi-paragraph long reasoning (3+ paragraphs, 300+ chars)."""
    paragraphs = [p for p in text.strip().split("\n\n") if p.strip()]
    return len(paragraphs) >= 3 and len(text) > 300


def check_appreciation_style(text: str) -> bool:
    """Check if text contains appreciation-style writing patterns."""
    return any(pattern.search(text) for pattern in APPRECIATION_PATTERNS)


def find_semantic_opposites(text_a: str, text_b: str) -> list[str]:
    """Find antonym conflicts between two texts.

    Returns list of antonym pair descriptions found, e.g. ["茂盛 vs 衰败"].
    """
    conflicts: list[str] = []
    for a, b in ANTONYM_PAIRS:
        a_in_a = a in text_a
        b_in_b = b in text_b
        b_in_a = b in text_a
        a_in_b = a in text_b
        if (a_in_a and b_in_b) or (b_in_a and a_in_b):
            conflicts.append(f"{a} vs {b}")
    return conflicts


def extract_teacher_content_fields(
    sample: dict[str, Any], path: str = "", skip_prefixes: tuple[str, ...] = SKIP_FIELD_CHECK_PREFIXES
) -> list[tuple[str, str]]:
    """Recursively extract (field_path, value) pairs for teacher-generated string fields.

    Skips fields under skip_prefixes (task, candidate_answer) since those are input data.
    """
    results: list[tuple[str, str]] = []

    # Skip entire subtrees that are input data
    current_key = path.split(".")[0] if path else ""
    if current_key in skip_prefixes:
        return results

    if isinstance(sample, str):
        results.append((path, sample))
    elif isinstance(sample, dict):
        for key, value in sample.items():
            key_path = f"{path}.{key}" if path else key
            if key in skip_prefixes:
                continue
            if isinstance(value, str):
                results.append((key_path, value))
            elif isinstance(value, (dict, list)):
                results.extend(extract_teacher_content_fields(value, key_path, skip_prefixes))
    elif isinstance(sample, list):
        for i, value in enumerate(sample):
            item_path = f"{path}[{i}]"
            if isinstance(value, str):
                results.append((item_path, value))
            elif isinstance(value, (dict, list)):
                results.extend(extract_teacher_content_fields(value, item_path, skip_prefixes))

    return results


# ============================================================
# Filtering class
# ============================================================

class TeacherDataFilter:
    """Applies Section 6 filtering rules to teacher-generated data."""

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    # ----------------------------------------------------------
    # Main processing
    # ----------------------------------------------------------

    def process(self, input_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
        """Run the full filter pipeline. Returns statistics dict."""
        raw_lines = input_path.read_text(encoding="utf-8").strip().split("\n")
        non_empty_lines = [line for line in raw_lines if line.strip()]
        total = len(non_empty_lines)

        json_parse_errors = 0
        passed: list[tuple[dict[str, Any], list[tuple[str, str, str]]]] = []
        filtered_reasons: Counter[str] = Counter()
        human_review_reasons: Counter[str] = Counter()

        for line in non_empty_lines:
            sample = self._parse_json_line(line)
            if sample is None:
                json_parse_errors += 1
                filtered_reasons["json_parse_error"] += 1
                continue

            issues = self._check_sample(sample)

            filter_issues = [(s, c, d) for s, c, d in issues if s == "filter"]
            review_issues = [(s, c, d) for s, c, d in issues if s == "human_review"]

            for _, code, _ in filter_issues:
                filtered_reasons[code] += 1

            for _, code, _ in review_issues:
                human_review_reasons[code] += 1

            if filter_issues:
                continue

            if self.strict and review_issues:
                filtered_reasons["strict_mode_human_review"] += 1
                continue

            passed.append((sample, review_issues))

        # 6.5 Deduplication
        dedup_removed = 0
        seen_keys: set[tuple] = set()
        final_passed: list[dict[str, Any]] = []

        for sample, review_issues in passed:
            key = self._dedup_key(sample)
            if key in seen_keys:
                dedup_removed += 1
                continue
            seen_keys.add(key)

            # For samples with human-review issues, attach quality_flags
            if review_issues:
                sample = self._attach_review_flags(sample, review_issues)
            final_passed.append(sample)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in final_passed:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Build report
        report = self._build_report(
            input_path=input_path,
            output_path=output_path,
            total=total,
            json_parse_errors=json_parse_errors,
            passed_before_dedup=len(passed),
            dedup_removed=dedup_removed,
            final_passed=len(final_passed),
            filtered_reasons=filtered_reasons,
            human_review_reasons=human_review_reasons,
        )

        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return report

    # ----------------------------------------------------------
    # Per-sample checks
    # ----------------------------------------------------------

    def _check_sample(self, sample: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Run all Section 6 checks on a single sample.

        Returns list of (severity, code, detail) tuples.
        severity is 'filter' or 'human_review'.
        """
        issues: list[tuple[str, str, str]] = []

        # 6.1 Field legality
        issues.extend(self._check_6_1(sample))
        # If record_type is invalid, we cannot proceed with record-type-specific checks
        record_type = sample.get("record_type")
        if record_type not in VALID_RECORD_TYPES:
            return issues

        # Extract task for coverage checks
        task: dict[str, Any] | None = sample.get("task")
        if not isinstance(task, dict):
            issues.append(("filter", "missing_task", "task field is missing or not an object"))
            return issues

        # 6.2 Coverage & option legality
        issues.extend(self._check_6_2(sample, task))

        # 6.3 Length & style
        issues.extend(self._check_6_3(sample))

        # 6.4 Evidence & answer consistency
        issues.extend(self._check_6_4(sample, task))

        return issues

    # ----------------------------------------------------------
    # 6.1 JSON & Field Legality
    # ----------------------------------------------------------

    def _check_6_1(self, sample: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Section 6.1: JSON & Field Legality."""
        issues: list[tuple[str, str, str]] = []

        # record_type check
        record_type = sample.get("record_type")
        if record_type not in VALID_RECORD_TYPES:
            issues.append(("filter", "invalid_record_type",
                           f"record_type={record_type!r}, expected short_evidence or teacher_critique"))
            # Don't return early; accumulate as many issues as possible

        # idx must match task.idx
        idx = sample.get("idx")
        task_idx = sample.get("task", {}).get("idx")
        if idx is None or task_idx is None:
            issues.append(("filter", "missing_idx", "idx or task.idx is missing"))
        elif idx != task_idx:
            issues.append(("filter", "idx_mismatch",
                           f"sample.idx={idx} != task.idx={task_idx}"))

        # Required fields based on record_type
        if record_type == "teacher_critique":
            tc_required = [
                "task", "candidate_answer", "critique",
                "correction_evidence", "corrected_sentiment", "corrected_answer",
            ]
            for field in tc_required:
                if field not in sample:
                    issues.append(("filter", "missing_field",
                                   f"Required field {field!r} missing for teacher_critique"))
        else:
            # short_evidence or unknown type
            se_required = ["task", "evidence", "sentiment", "draft_answer"]
            for field in se_required:
                if field not in sample:
                    issues.append(("filter", "missing_field",
                                   f"Required field {field!r} missing"))

        # Forbid free CoT field names (recursive)
        cot_paths = self._find_cot_field_names(sample)
        for path in cot_paths:
            issues.append(("filter", "forbidden_cot_field",
                           f"Contains forbidden CoT field at: {path}"))

        # Check all teacher-generated string fields for markdown, prompt residue, long reasoning
        for field_path, value in extract_teacher_content_fields(sample):
            if has_markdown_code_blocks(value):
                issues.append(("filter", "markdown_code_block",
                               f"Field {field_path} contains markdown code block"))
            if has_prompt_residue(value):
                issues.append(("filter", "prompt_residue",
                               f"Field {field_path} contains prompt residue"))
            if has_multi_paragraph_long_reasoning(value):
                issues.append(("filter", "long_reasoning",
                               f"Field {field_path} has multi-paragraph long reasoning"))

        return issues

    # ----------------------------------------------------------
    # 6.2 Coverage & Option Legality
    # ----------------------------------------------------------

    def _check_6_2(self, sample: dict[str, Any], task: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Section 6.2: Coverage & Option Legality."""
        issues: list[tuple[str, str, str]] = []
        record_type = sample.get("record_type")

        qa_words = unique_preserve_order(task.get("qa_words", []))
        qa_sents = unique_preserve_order(task.get("qa_sents", []))
        choose = task.get("choose", {})

        # Determine which answer fields to check based on record_type
        if record_type == "short_evidence":
            draft = sample.get("draft_answer", {})
            ans_words = draft.get("ans_qa_words", {}) if isinstance(draft, dict) else {}
            ans_sents = draft.get("ans_qa_sents", {}) if isinstance(draft, dict) else {}
        elif record_type == "teacher_critique":
            corrected = sample.get("corrected_answer", {})
            ans_words = corrected.get("ans_qa_words", {}) if isinstance(corrected, dict) else {}
            ans_sents = corrected.get("ans_qa_sents", {}) if isinstance(corrected, dict) else {}
        else:
            ans_words = {}
            ans_sents = {}

        # Coverage: ans_qa_words keys must equal dedup qa_words
        if not isinstance(ans_words, dict):
            issues.append(("filter", "invalid_ans_qa_words",
                           "ans_qa_words is not an object"))
        else:
            word_keys = set(ans_words.keys())
            required_words = set(qa_words)
            missing_words = required_words - word_keys
            extra_words = word_keys - required_words
            if missing_words:
                issues.append(("filter", "missing_ans_qa_words",
                               f"Missing keys: {sorted(missing_words)}"))
            if extra_words:
                issues.append(("filter", "extra_ans_qa_words",
                               f"Unexpected keys: {sorted(extra_words)}"))

        # Coverage: ans_qa_sents keys must equal dedup qa_sents
        if not isinstance(ans_sents, dict):
            issues.append(("filter", "invalid_ans_qa_sents",
                           "ans_qa_sents is not an object"))
        else:
            sent_keys = set(ans_sents.keys())
            required_sents = set(qa_sents)
            missing_sents = required_sents - sent_keys
            extra_sents = sent_keys - required_sents
            if missing_sents:
                issues.append(("filter", "missing_ans_qa_sents",
                               f"Missing keys: {sorted(missing_sents)}"))
            if extra_sents:
                issues.append(("filter", "extra_ans_qa_sents",
                               f"Unexpected keys: {sorted(extra_sents)}"))

        # For training samples (choose is empty), draft_answer must not contain choose_id
        if record_type == "short_evidence":
            draft = sample.get("draft_answer", {})
            if isinstance(draft, dict):
                if not choose and "choose_id" in draft:
                    issues.append(("filter", "unexpected_choose_id",
                                   "Training sample (choose={}) has choose_id in draft_answer"))
                if choose and "choose_id" in draft:
                    choose_id = draft["choose_id"]
                    if choose_id not in choose:
                        issues.append(("filter", "invalid_choose_id",
                                       f"choose_id={choose_id!r} not in choose keys {set(choose)}"))

        # For teacher_critique: corrected_answer must not contain choose_id
        if record_type == "teacher_critique":
            corrected = sample.get("corrected_answer", {})
            if isinstance(corrected, dict) and "choose_id" in corrected:
                issues.append(("filter", "unexpected_choose_id",
                               "teacher_critique corrected_answer should not contain choose_id"))

        # sentiment.primary must be in controlled vocabulary
        sentiment = sample.get("corrected_sentiment", {}) if record_type == "teacher_critique" else sample.get("sentiment", {})
        if isinstance(sentiment, dict):
            primary = sentiment.get("primary")
            if primary is None:
                issues.append(("filter", "missing_sentiment_primary",
                               f"{'corrected_' if record_type == 'teacher_critique' else ''}sentiment.primary is missing"))
            elif primary not in CONTROLLED_VOCABULARY:
                issues.append(("filter", "primary_not_in_vocab",
                               f"{'corrected_' if record_type == 'teacher_critique' else ''}sentiment.primary={primary!r} not in controlled vocabulary"))

            # sentiment.secondary: labels not in vocab -> human_review
            secondary = sentiment.get("secondary", [])
            if isinstance(secondary, list):
                for label in secondary:
                    if label not in CONTROLLED_VOCABULARY:
                        issues.append(("human_review", "secondary_not_in_vocab",
                                       f"{'corrected_' if record_type == 'teacher_critique' else ''}sentiment.secondary label {label!r} not in controlled vocabulary"))
        elif record_type != "teacher_critique":
            # teacher_critique doesn't require top-level sentiment
            issues.append(("filter", "invalid_sentiment", "sentiment is not an object"))

        return issues

    # ----------------------------------------------------------
    # 6.3 Length & Style
    # ----------------------------------------------------------

    def _check_6_3(self, sample: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Section 6.3: Length & Style."""
        issues: list[tuple[str, str, str]] = []
        record_type = sample.get("record_type")

        # Determine which answer fields to check based on record_type
        if record_type == "short_evidence":
            draft = sample.get("draft_answer", {})
            ans_words = draft.get("ans_qa_words", {}) if isinstance(draft, dict) else {}
            ans_sents = draft.get("ans_qa_sents", {}) if isinstance(draft, dict) else {}
        elif record_type == "teacher_critique":
            corrected = sample.get("corrected_answer", {})
            ans_words = corrected.get("ans_qa_words", {}) if isinstance(corrected, dict) else {}
            ans_sents = corrected.get("ans_qa_sents", {}) if isinstance(corrected, dict) else {}
        else:
            ans_words = {}
            ans_sents = {}

        # Word answer lengths (6.3.1)
        if isinstance(ans_words, dict):
            for word, answer in ans_words.items():
                if not isinstance(answer, str):
                    continue
                cjk_count = count_cjk(answer)
                if cjk_count > 80:
                    issues.append(("filter", "word_answer_too_long",
                                   f"ans_qa_words[{word!r}] has {cjk_count} CJK chars (>80)"))
                elif cjk_count > 50:
                    issues.append(("human_review", "word_answer_over_50",
                                   f"ans_qa_words[{word!r}] has {cjk_count} CJK chars (>50)"))
                # Appreciation-style check
                if check_appreciation_style(answer):
                    issues.append(("filter", "appreciation_style",
                                   f"ans_qa_words[{word!r}] contains appreciation-style writing"))

        # Sentence translation lengths (6.3.2)
        if isinstance(ans_sents, dict):
            for sent, translation in ans_sents.items():
                if not isinstance(translation, str):
                    continue
                cjk_count = count_cjk(translation)
                if cjk_count > 180:
                    issues.append(("filter", "sent_translation_too_long",
                                   f"ans_qa_sents[{sent!r}] has {cjk_count} CJK chars (>180)"))
                elif cjk_count > 120:
                    issues.append(("human_review", "sent_translation_over_120",
                                   f"ans_qa_sents[{sent!r}] has {cjk_count} CJK chars (>120)"))
                if check_appreciation_style(translation):
                    issues.append(("filter", "appreciation_style",
                                   f"ans_qa_sents[{sent!r}] contains appreciation-style writing"))

        # rationale/comment/support lengths (6.3.3)
        issues.extend(self._check_field_length_rationale(sample, "rationale", 60, 100))
        issues.extend(self._check_field_length_rationale(sample, "comment", 60, 100))
        issues.extend(self._check_field_length_rationale(sample, "support", 60, 100))

        # sentiment.rationale length (6.3.4)
        # For teacher_critique, use corrected_sentiment instead of sentiment
        sentiment_key = "corrected_sentiment" if record_type == "teacher_critique" else "sentiment"
        sentiment = sample.get(sentiment_key, {})
        if isinstance(sentiment, dict):
            rationale = sentiment.get("rationale", "")
            if isinstance(rationale, str):
                cjk_count = count_cjk(rationale)
                if cjk_count > 120:
                    issues.append(("filter", "sentiment_rationale_too_long",
                                   f"{sentiment_key}.rationale has {cjk_count} CJK chars (>120)"))
                elif cjk_count > 80:
                    issues.append(("human_review", "sentiment_rationale_over_80",
                                   f"{sentiment_key}.rationale has {cjk_count} CJK chars (>80)"))
                if check_appreciation_style(rationale):
                    issues.append(("filter", "appreciation_style",
                                   f"{sentiment_key}.rationale contains appreciation-style writing"))

        # Check teacher-generated content fields for appreciation-style writing
        for field_path, value in extract_teacher_content_fields(sample):
            if check_appreciation_style(value):
                issues.append(("filter", "appreciation_style",
                               f"Field {field_path} contains appreciation-style writing"))

        return issues

    def _check_field_length_rationale(
        self, sample: dict[str, Any], field_name: str,
        review_threshold: int, filter_threshold: int,
    ) -> list[tuple[str, str, str]]:
        """Check length of rationale/comment/support fields in evidence and critique."""
        issues: list[tuple[str, str, str]] = []

        # Check evidence.words[*].{field_name}
        evidence = sample.get("evidence", {})
        if isinstance(evidence, dict):
            words = evidence.get("words", {})
            if isinstance(words, dict):
                for word_key, word_val in words.items():
                    if isinstance(word_val, dict):
                        text = word_val.get(field_name, "")
                        if isinstance(text, str):
                            cjk = count_cjk(text)
                            if cjk > filter_threshold:
                                issues.append(("filter", f"{field_name}_too_long",
                                               f"evidence.words[{word_key!r}].{field_name} "
                                               f"has {cjk} CJK chars (>{filter_threshold})"))
                            elif cjk > review_threshold:
                                issues.append(("human_review", f"{field_name}_over_{review_threshold}",
                                               f"evidence.words[{word_key!r}].{field_name} "
                                               f"has {cjk} CJK chars (>{review_threshold})"))

            sentences = evidence.get("sentences", {})
            if isinstance(sentences, dict):
                for sent_key, sent_val in sentences.items():
                    if isinstance(sent_val, dict):
                        text = sent_val.get(field_name, "")
                        if isinstance(text, str):
                            cjk = count_cjk(text)
                            if cjk > filter_threshold:
                                issues.append(("filter", f"{field_name}_too_long",
                                               f"evidence.sentences[{sent_key!r}].{field_name} "
                                               f"has {cjk} CJK chars (>{filter_threshold})"))
                            elif cjk > review_threshold:
                                issues.append(("human_review", f"{field_name}_over_{review_threshold}",
                                               f"evidence.sentences[{sent_key!r}].{field_name} "
                                               f"has {cjk} CJK chars (>{review_threshold})"))

        # Check teacher_critique correction_evidence
        if sample.get("record_type") == "teacher_critique":
            correction_evidence = sample.get("correction_evidence", {})
            if isinstance(correction_evidence, dict):
                c_words = correction_evidence.get("words", {})
                if isinstance(c_words, dict):
                    for word_key, word_val in c_words.items():
                        if isinstance(word_val, str):
                            cjk = count_cjk(word_val)
                            if cjk > filter_threshold:
                                issues.append(("filter", f"{field_name}_too_long",
                                               f"correction_evidence.words[{word_key!r}] "
                                               f"has {cjk} CJK chars"))
                            elif cjk > review_threshold:
                                issues.append(("human_review", f"{field_name}_over_{review_threshold}",
                                               f"correction_evidence.words[{word_key!r}] "
                                               f"has {cjk} CJK chars"))
                c_sents = correction_evidence.get("sentences", {})
                if isinstance(c_sents, dict):
                    for sent_key, sent_val in c_sents.items():
                        if isinstance(sent_val, str):
                            cjk = count_cjk(sent_val)
                            if cjk > filter_threshold:
                                issues.append(("filter", f"{field_name}_too_long",
                                               f"correction_evidence.sentences[{sent_key!r}] "
                                               f"has {cjk} CJK chars"))
                            elif cjk > review_threshold:
                                issues.append(("human_review", f"{field_name}_over_{review_threshold}",
                                               f"correction_evidence.sentences[{sent_key!r}] "
                                               f"has {cjk} CJK chars"))

        # Check critique comment fields
        critique = sample.get("critique", {})
        if isinstance(critique, dict):
            for err_type in ("word_errors", "sentence_errors"):
                errors = critique.get(err_type, [])
                if isinstance(errors, list):
                    for err in errors:
                        if isinstance(err, dict):
                            comment = err.get("comment", "")
                            if isinstance(comment, str):
                                cjk = count_cjk(comment)
                                if cjk > filter_threshold:
                                    issues.append(("filter", f"comment_too_long",
                                                   f"critique.{err_type}.comment "
                                                   f"has {cjk} CJK chars (>{filter_threshold})"))
                                elif cjk > review_threshold:
                                    issues.append(("human_review", f"comment_over_{review_threshold}",
                                                   f"critique.{err_type}.comment "
                                                   f"has {cjk} CJK chars (>{review_threshold})"))

        return issues

    # ----------------------------------------------------------
    # 6.4 Evidence & Answer Consistency
    # ----------------------------------------------------------

    def _check_6_4(self, sample: dict[str, Any], task: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Section 6.4: Evidence & Answer Consistency."""
        issues: list[tuple[str, str, str]] = []
        record_type = sample.get("record_type")

        # Select fields based on record type
        if record_type == "teacher_critique":
            evidence = sample.get("correction_evidence", {})
            sentiment_key = "corrected_sentiment"
            sentiment = sample.get(sentiment_key, {})
            corrected = sample.get("corrected_answer", {})
            ans_words = corrected.get("ans_qa_words", {}) if isinstance(corrected, dict) else {}
            ans_sents = corrected.get("ans_qa_sents", {}) if isinstance(corrected, dict) else {}
        else:
            evidence = sample.get("evidence", {})
            sentiment_key = "sentiment"
            sentiment = sample.get(sentiment_key, {})
            draft = sample.get("draft_answer", {})
            ans_words = draft.get("ans_qa_words", {}) if isinstance(draft, dict) else {}
            ans_sents = draft.get("ans_qa_sents", {}) if isinstance(draft, dict) else {}

        if not isinstance(evidence, dict):
            return issues

        # 6.4.1 evidence.words[*].meaning vs draft_answer.ans_qa_words[*]
        ev_words = evidence.get("words", {})
        if isinstance(ev_words, dict) and isinstance(ans_words, dict):
            for word_key in set(ev_words.keys()) & set(ans_words.keys()):
                ev_word = ev_words[word_key]
                ans_val = ans_words[word_key]
                if isinstance(ev_word, dict) and isinstance(ans_val, str):
                    meaning = ev_word.get("meaning", "")
                    if isinstance(meaning, str):
                        conflicts = find_semantic_opposites(meaning, ans_val)
                        if conflicts:
                            issues.append(("filter", "semantic_opposite",
                                           f"{'correction_' if record_type == 'teacher_critique' else ''}"
                                           f"evidence.words[{word_key!r}].meaning conflicts with "
                                           f"ans_qa_words[{word_key!r}]: {conflicts}"))

        # 6.4.2 sentiment.primary consistency with evidence.emotion
        ev_emotion = evidence.get("emotion", [])
        if isinstance(sentiment, dict) and isinstance(ev_emotion, list):
            primary = sentiment.get("primary", "")
            if isinstance(primary, str) and primary in CONTROLLED_VOCABULARY:
                if primary in POSITIVE_SENTIMENT_LABELS:
                    emotion_text = " ".join(str(e) for e in ev_emotion if isinstance(e, str))
                    neg_found = [kw for kw in NEGATIVE_KEYWORDS if kw in emotion_text]
                    pos_found = [kw for kw in POSITIVE_KEYWORDS if kw in emotion_text]
                    if neg_found and not pos_found:
                        issues.append(("human_review", "emotion_sentiment_mismatch",
                                       f"{sentiment_key}.primary={primary!r} (positive) but "
                                       f"evidence.emotion contains negative keywords: {neg_found}"))
                elif primary in NEGATIVE_SENTIMENT_LABELS:
                    emotion_text = " ".join(str(e) for e in ev_emotion if isinstance(e, str))
                    pos_found = [kw for kw in POSITIVE_KEYWORDS if kw in emotion_text]
                    neg_found = [kw for kw in NEGATIVE_KEYWORDS if kw in emotion_text]
                    if pos_found and not neg_found:
                        issues.append(("human_review", "emotion_sentiment_mismatch",
                                       f"{sentiment_key}.primary={primary!r} (negative) but "
                                       f"evidence.emotion contains positive keywords: {pos_found}"))

        # 6.4.3 sentiment.rationale must cite specific words/imagery/sentences
        if isinstance(sentiment, dict):
            rationale = sentiment.get("rationale", "")
            if not isinstance(rationale, str) or not rationale.strip():
                issues.append(("human_review", "empty_sentiment_rationale",
                               f"{sentiment_key}.rationale is empty"))
            elif count_cjk(rationale) < 8:
                issues.append(("human_review", "generic_sentiment_rationale",
                               f"{sentiment_key}.rationale too short ({count_cjk(rationale)} CJK chars)"))
            else:
                poem_content = task.get("content", "")
                if isinstance(poem_content, str):
                    poem_chars = set(poem_content)
                    rationale_chars = set(rationale)
                    shared_chars = poem_chars & rationale_chars
                    chinese_chars_in_rationale = {c for c in rationale if "一" <= c <= "鿿"}
                    if len(shared_chars & chinese_chars_in_rationale) < 3:
                        issues.append(("human_review", "generic_sentiment_rationale",
                                       f"{sentiment_key}.rationale does not cite specific poem content"))

        # If sentiment.primary is "其他", ensure needs_human_review in quality_flags
        if isinstance(sentiment, dict):
            primary = sentiment.get("primary", "")
            quality_flags = sample.get("quality_flags", [])
            if isinstance(quality_flags, list) and primary == "其他":
                if "needs_human_review" not in quality_flags:
                    issues.append(("human_review", "needs_human_review_missing",
                                   f"{sentiment_key}.primary is '其他' but quality_flags missing needs_human_review"))

        # 6.4.6 teacher_critique: emotion_error.primary_error_type == "correct" consistency
        if record_type == "teacher_critique":
            critique = sample.get("critique", {})
            if isinstance(critique, dict):
                emotion_error = critique.get("emotion_error", {})
                if isinstance(emotion_error, dict):
                    pet = emotion_error.get("primary_error_type")
                    candidate_primary = emotion_error.get("candidate_primary")
                    correct_primary = emotion_error.get("correct_primary")
                    corrected_sentiment = sample.get("corrected_sentiment", {})
                    if isinstance(corrected_sentiment, dict):
                        cs_primary = corrected_sentiment.get("primary")
                        if pet == "correct" and candidate_primary != correct_primary:
                            issues.append(("filter", "critique_emotion_inconsistent",
                                           f"primary_error_type='correct' but candidate_primary="
                                           f"{candidate_primary!r} != correct_primary={correct_primary!r}"))

        return issues

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _parse_json_line(self, line: str) -> dict[str, Any] | None:
        """Parse a single JSONL line. Returns None on failure."""
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        return obj

    def _find_cot_field_names(self, obj: Any, path: str = "") -> list[str]:
        """Recursively find forbidden CoT field names."""
        paths: list[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else str(key)
                if isinstance(key, str) and key.lower() in FORBIDDEN_COT_FIELDS:
                    paths.append(current_path)
                paths.extend(self._find_cot_field_names(value, current_path))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                paths.extend(self._find_cot_field_names(value, f"{path}[{i}]"))
        return paths

    def _dedup_key(self, sample: dict[str, Any]) -> tuple:
        """Build a hashable dedup key from sample.

        Based on Section 6.5: same idx + record_type + draft_answer + sentiment.
        For teacher_critique, uses corrected_sentiment instead of sentiment.
        """
        idx = sample.get("idx")
        record_type = sample.get("record_type")

        if record_type == "short_evidence":
            draft = sample.get("draft_answer", {})
            sentiment = sample.get("sentiment", {})
        elif record_type == "teacher_critique":
            draft = sample.get("candidate_answer", {}).get("draft_answer", {})
            sentiment = sample.get("corrected_sentiment", {})
        else:
            draft = sample.get("draft_answer", {})
            sentiment = sample.get("sentiment", {})

        draft_str = json.dumps(draft, sort_keys=True, ensure_ascii=False)
        sentiment_str = json.dumps(sentiment, sort_keys=True, ensure_ascii=False)

        return (idx, record_type, draft_str, sentiment_str)

    def _attach_review_flags(
        self, sample: dict[str, Any], review_issues: list[tuple[str, str, str]]
    ) -> dict[str, Any]:
        """Attach quality_flags based on human-review issues.

        Returns a new dict (shallow copy) with updated quality_flags.
        """
        sample = dict(sample)
        quality_flags = list(sample.get("quality_flags", []) or [])

        codes = {code for _, code, _ in review_issues}

        if "generic_sentiment_rationale" in codes:
            if "low_confidence_emotion" not in quality_flags:
                quality_flags.append("low_confidence_emotion")

        if "emotion_sentiment_mismatch" in codes:
            if "needs_human_review" not in quality_flags:
                quality_flags.append("needs_human_review")

        if "needs_human_review_missing" in codes:
            if "needs_human_review" not in quality_flags:
                quality_flags.append("needs_human_review")

        if "secondary_not_in_vocab" in codes:
            if "needs_human_review" not in quality_flags:
                quality_flags.append("needs_human_review")

        sample["quality_flags"] = quality_flags
        return sample

    # ----------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------

    def _build_report(
        self,
        input_path: Path,
        output_path: Path,
        total: int,
        json_parse_errors: int,
        passed_before_dedup: int,
        dedup_removed: int,
        final_passed: int,
        filtered_reasons: Counter[str],
        human_review_reasons: Counter[str],
    ) -> dict[str, Any]:
        """Build the statistics report dict.

        Inventory reconciliation:
          total = json_parse_errors + filtered_by_rules + passed_before_dedup
          passed_before_dedup = dedup_removed + final_passed
          filtered_by_rules = total - json_parse_errors - passed_before_dedup
        """
        entries_after_parse = total - json_parse_errors
        filtered_by_rules = entries_after_parse - passed_before_dedup

        # Separate out strict-mode discards from other filter reasons
        strict_discards = filtered_reasons.pop("strict_mode_human_review", 0)

        return {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "strict_mode": self.strict,
            "total_samples": total,
            "json_parse_errors": json_parse_errors,
            "entries_entering_checks": entries_after_parse,
            "filtered_by_rules": filtered_by_rules,
            "filter_reasons": dict(filtered_reasons.most_common()),
            "human_review_flagged_count": human_review_reasons.total(),
            "human_review_reasons": dict(human_review_reasons.most_common()),
            "strict_human_review_discarded": strict_discards,
            "passed_before_dedup": passed_before_dedup,
            "dedup_removed": dedup_removed,
            "passed": final_passed,
        }


