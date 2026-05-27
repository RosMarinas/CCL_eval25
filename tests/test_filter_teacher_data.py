"""Comprehensive integration tests for filter_teacher_data.py"""

import json
import tempfile
from pathlib import Path
from src.cli.filter_teacher_data import (
    TeacherDataFilter,
    count_cjk,
    unique_preserve_order,
    CONTROLLED_VOCABULARY,
    FORBIDDEN_COT_FIELDS,
    has_markdown_code_blocks,
    has_prompt_residue,
    extract_teacher_content_fields,
)


def make_valid_base():
    return {
        "record_type": "short_evidence",
        "idx": 0,
        "task": {
            "idx": 0,
            "title": "李端公",
            "author": "卢纶",
            "content": "故关衰草遍，离别自堪悲。",
            "qa_words": ["衰草", "故关"],
            "qa_sents": ["故关衰草遍，离别自堪悲"],
            "choose": {},
        },
        "evidence": {
            "words": {
                "衰草": {"meaning": "枯黄的草", "text_clue": "故关衰草遍", "rationale": "烘托荒凉离别的氛围"},
                "故关": {"meaning": "旧日关塞", "text_clue": "故关衰草遍", "rationale": "离别之地"},
            },
            "sentences": {
                "故关衰草遍，离别自堪悲": {
                    "translation": "旧关一带长满枯草，分别本就令人悲伤",
                    "key_images": ["故关", "衰草", "离别"],
                    "rationale": "以衰败景象写离别之悲",
                }
            },
            "emotion": ["衰草、离别、悲等词指向伤感"],
        },
        "sentiment": {
            "primary": "惜别感伤",
            "secondary": ["羁旅思归"],
            "rationale": "衰草、离别、悲等意象共同指向离别之伤感",
        },
        "draft_answer": {
            "ans_qa_words": {
                "衰草": "枯黄的草，烘托荒凉离别的氛围",
                "故关": "旧日关塞，也含故乡关隘之意",
            },
            "ans_qa_sents": {
                "故关衰草遍，离别自堪悲": "旧关一带长满枯草，分别本就令人悲伤"
            },
        },
        "quality_flags": [],
    }


def run_test(samples, strict=False, desc=""):
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "input.jsonl"
        out = Path(tmpdir) / "output.jsonl"
        rep = Path(tmpdir) / "report.json"

        with open(inp, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        filterer = TeacherDataFilter(strict=strict)
        stats = filterer.process(inp, out, rep)

        with open(out, "r", encoding="utf-8") as f:
            output_lines = [l.strip() for l in f if l.strip()]

        print(
            f"{desc}: total={stats['total_samples']} passed={stats['passed']} "
            f"filtered={stats['filtered_by_rules']} parse_err={stats['json_parse_errors']} "
            f"dedup={stats['dedup_removed']}"
        )
        return stats, output_lines


# --- Unit tests ---

def test_helpers():
    assert count_cjk("abc") == 0
    assert count_cjk("枯黄的草") == 4
    assert count_cjk("枯黄的草abc123") == 4
    print("PASS: count_cjk")

    assert unique_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
    assert unique_preserve_order([]) == []
    print("PASS: unique_preserve_order")

    assert "惜别感伤" in CONTROLLED_VOCABULARY
    assert "其他" in CONTROLLED_VOCABULARY
    assert "random_label" not in CONTROLLED_VOCABULARY
    assert len(CONTROLLED_VOCABULARY) == 25
    print("PASS: controlled vocabulary")

    assert "reasoning" in FORBIDDEN_COT_FIELDS
    assert "cot" in FORBIDDEN_COT_FIELDS
    print("PASS: forbidden CoT fields")

    assert has_markdown_code_blocks("some text ```code``` more text") is True
    assert has_markdown_code_blocks("no code blocks here") is False
    print("PASS: has_markdown_code_blocks")

    assert has_prompt_residue("你是古诗词理解任务的教师模型") is True
    assert has_prompt_residue("正常文本") is False
    print("PASS: has_prompt_residue")

    # Test extract_teacher_content_fields skips task subtree
    sample = {
        "record_type": "short_evidence",
        "task": {"content": "你是input", "idx": 0},
        "evidence": {"emotion": ["正常文本"]},
    }
    fields = extract_teacher_content_fields(sample)
    task_fields = [p for p, _ in fields if p.startswith("task")]
    assert len(task_fields) == 0, f"task fields not skipped: {task_fields}"
    print("PASS: extract_teacher_content_fields skips task")


def test_valid_sample_passes():
    stats, _ = run_test([make_valid_base()], desc="valid sample")
    assert stats["total_samples"] == 1
    assert stats["passed"] == 1
    assert stats["filtered_by_rules"] == 0
    assert stats["dedup_removed"] == 0
    print("PASS: valid sample passes")


def test_invalid_record_type():
    s = make_valid_base()
    s["record_type"] = "invalid_type"
    stats, _ = run_test([s], desc="invalid record_type")
    assert stats["filtered_by_rules"] == 1
    print("PASS: invalid record_type filtered")


def test_missing_required_field():
    s = make_valid_base()
    del s["evidence"]
    stats, _ = run_test([s], desc="missing evidence")
    assert stats["filtered_by_rules"] == 1
    print("PASS: missing required field filtered")


def test_forbidden_cot_field():
    s = make_valid_base()
    s["reasoning"] = "some reasoning text"
    stats, _ = run_test([s], desc="forbidden CoT field")
    assert stats["filtered_by_rules"] == 1
    print("PASS: forbidden CoT field filtered")


def test_idx_mismatch():
    s = make_valid_base()
    s["idx"] = 99
    stats, _ = run_test([s], desc="idx mismatch")
    assert stats["filtered_by_rules"] == 1
    print("PASS: idx mismatch filtered")


def test_markdown_code_block():
    s = make_valid_base()
    s["evidence"]["words"]["衰草"]["meaning"] = "START_MD " + "```" + " code " + "```" + " END_MD"
    stats, _ = run_test([s], desc="markdown code block")
    assert stats["filtered_by_rules"] == 1, f"Expected filtered=1, got filtered={stats['filtered']}"
    print("PASS: markdown code block filtered")


def test_prompt_residue():
    s = make_valid_base()
    s["sentiment"]["rationale"] = "你是古诗词理解任务的教师模型"
    stats, _ = run_test([s], desc="prompt residue")
    assert stats["filtered_by_rules"] == 1
    print("PASS: prompt residue filtered")


def test_coverage_missing_word_key():
    s = make_valid_base()
    s["draft_answer"]["ans_qa_words"] = {"衰草": "枯黄的草"}
    stats, _ = run_test([s], desc="missing word key")
    assert stats["filtered_by_rules"] == 1
    print("PASS: missing word key filtered")


def test_coverage_extra_word_key():
    s = make_valid_base()
    s["draft_answer"]["ans_qa_words"]["extra"] = "extra word"
    stats, _ = run_test([s], desc="extra word key")
    assert stats["filtered_by_rules"] == 1
    print("PASS: extra word key filtered")


def test_primary_not_in_vocab():
    s = make_valid_base()
    s["sentiment"]["primary"] = "快乐无比"
    stats, _ = run_test([s], desc="primary not in vocab")
    assert stats["filtered_by_rules"] == 1
    print("PASS: primary not in vocab filtered")


def test_secondary_not_in_vocab_human_review():
    s = make_valid_base()
    s["sentiment"]["secondary"] = ["快乐无比"]
    stats, _ = run_test([s], desc="secondary not in vocab (human_review)")
    assert stats["filtered_by_rules"] == 0
    assert stats["passed"] == 1
    print("PASS: secondary not in vocab -> human review (not filtered)")


def test_word_answer_too_long_filter():
    s = make_valid_base()
    s["draft_answer"]["ans_qa_words"]["衰草"] = "枯" * 81
    stats, _ = run_test([s], desc="word answer >80 filter")
    assert stats["filtered_by_rules"] == 1
    print("PASS: word answer >80 filtered")


def test_word_answer_over_50_human_review():
    s = make_valid_base()
    s["draft_answer"]["ans_qa_words"]["衰草"] = "枯" * 55
    stats, _ = run_test([s], strict=False, desc="word answer >50 (human_review, non-strict)")
    assert stats["filtered_by_rules"] == 0
    print("PASS: word answer >50 -> human review (passes in non-strict)")


def test_word_answer_over_50_strict():
    s = make_valid_base()
    s["draft_answer"]["ans_qa_words"]["衰草"] = "枯" * 55
    stats, _ = run_test([s], strict=True, desc="word answer >50 (strict)")
    assert stats["strict_human_review_discarded"] == 1, \
        f"Expected strict_human_review_discarded=1, got {stats}"
    assert stats["passed"] == 0
    print("PASS: word answer >50 discarded in strict mode")


def test_sent_translation_too_long():
    s = make_valid_base()
    s["draft_answer"]["ans_qa_sents"]["故关衰草遍，离别自堪悲"] = "旧" * 181
    stats, _ = run_test([s], desc="sent translation >180")
    assert stats["filtered_by_rules"] == 1
    print("PASS: sent translation >180 filtered")


def test_appreciation_style():
    s = make_valid_base()
    s["sentiment"]["rationale"] = "这首诗表达了诗人对离别的感伤之情"
    stats, _ = run_test([s], desc="appreciation style")
    assert stats["filtered_by_rules"] == 1
    print("PASS: appreciation style filtered")


def test_dedup_identical():
    s1 = make_valid_base()
    s2 = make_valid_base()
    stats, lines = run_test([s1, s2], desc="duplicate dedup")
    assert stats["total_samples"] == 2
    assert stats["dedup_removed"] == 1
    assert stats["passed"] == 1
    assert len(lines) == 1
    print("PASS: duplicate dedup works")


def test_json_parse_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        inp = Path(tmpdir) / "input.jsonl"
        out = Path(tmpdir) / "output.jsonl"
        rep = Path(tmpdir) / "report.json"
        with open(inp, "w", encoding="utf-8") as f:
            f.write("not valid json\n")
        filterer = TeacherDataFilter()
        stats = filterer.process(inp, out, rep)
        assert stats["json_parse_errors"] == 1
        assert stats["passed"] == 0
    print("PASS: JSON parse error counted")


def test_valid_teacher_critique():
    tc = {
        "record_type": "teacher_critique",
        "idx": 0,
        "task": {
            "idx": 0,
            "title": "李端公",
            "author": "卢纶",
            "content": "故关衰草遍，离别自堪悲。",
            "qa_words": ["衰草", "故关"],
            "qa_sents": ["故关衰草遍，离别自堪悲"],
            "choose": {},
        },
        "candidate_answer": {
            "evidence": {"words": {}, "sentences": {}, "emotion": []},
            "sentiment": {"primary": "田园之乐", "secondary": [], "rationale": "some rationale"},
            "draft_answer": {"ans_qa_words": {}, "ans_qa_sents": {}},
        },
        "critique": {
            "word_errors": [],
            "sentence_errors": [],
            "emotion_error": {
                "candidate_primary": "田园之乐",
                "correct_primary": "惜别感伤",
                "primary_error_type": "wrong_label",
                "secondary_error_type": "correct",
                "rationale_error_type": "correct",
                "comment": "comment",
            },
        },
        "correction_evidence": {
            "words": {"衰草": "枯黄的草", "故关": "旧日关塞"},
            "sentences": {"故关衰草遍，离别自堪悲": "旧关一带长满枯草"},
        },
        "corrected_sentiment": {
            "primary": "惜别感伤",
            "secondary": ["羁旅思归"],
            "rationale": "衰草离别悲等意象指向伤感",
        },
        "corrected_answer": {
            "idx": 0,
            "ans_qa_words": {"衰草": "枯黄的草", "故关": "旧日关塞"},
            "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关一带长满枯草"},
        },
        "quality_flags": [],
    }
    stats, _ = run_test([tc], desc="valid teacher_critique")
    assert stats["passed"] == 1
    print("PASS: valid teacher_critique passes")


def test_teacher_critique_with_choose_id():
    tc = {
        "record_type": "teacher_critique",
        "idx": 0,
        "task": {
            "idx": 0,
            "title": "李端公",
            "author": "卢纶",
            "content": "故关衰草遍，离别自堪悲。",
            "qa_words": ["衰草"],
            "qa_sents": ["故关衰草遍，离别自堪悲"],
            "choose": {},
        },
        "candidate_answer": {
            "evidence": {"words": {}, "sentences": {}, "emotion": []},
            "sentiment": {"primary": "田园之乐", "secondary": [], "rationale": "some rationale"},
            "draft_answer": {"ans_qa_words": {}, "ans_qa_sents": {}},
        },
        "critique": {
            "word_errors": [],
            "sentence_errors": [],
            "emotion_error": {
                "candidate_primary": "田园之乐",
                "correct_primary": "惜别感伤",
                "primary_error_type": "wrong_label",
                "secondary_error_type": "correct",
                "rationale_error_type": "correct",
                "comment": "comment",
            },
        },
        "correction_evidence": {"words": {"衰草": "枯黄的草"}, "sentences": {}},
        "corrected_sentiment": {
            "primary": "惜别感伤",
            "secondary": ["羁旅思归"],
            "rationale": "衰草离别悲等意象指向伤感",
        },
        "corrected_answer": {
            "idx": 0,
            "ans_qa_words": {"衰草": "枯黄的草"},
            "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关一带长满枯草"},
            "choose_id": "A",
        },
        "quality_flags": [],
    }
    stats, _ = run_test([tc], desc="teacher_critique with choose_id")
    assert stats["filtered_by_rules"] == 1
    print("PASS: teacher_critique with choose_id in corrected_answer filtered")


def test_teacher_critique_emotion_inconsistent():
    tc = {
        "record_type": "teacher_critique",
        "idx": 0,
        "task": {
            "idx": 0,
            "title": "李端公",
            "author": "卢纶",
            "content": "故关衰草遍，离别自堪悲。",
            "qa_words": ["衰草"],
            "qa_sents": ["故关衰草遍，离别自堪悲"],
            "choose": {},
        },
        "candidate_answer": {
            "evidence": {"words": {}, "sentences": {}, "emotion": []},
            "sentiment": {"primary": "田园之乐", "secondary": [], "rationale": "some rationale"},
            "draft_answer": {"ans_qa_words": {}, "ans_qa_sents": {}},
        },
        "critique": {
            "word_errors": [],
            "sentence_errors": [],
            "emotion_error": {
                "candidate_primary": "田园之乐",
                "correct_primary": "惜别感伤",
                "primary_error_type": "correct",
                "secondary_error_type": "correct",
                "rationale_error_type": "correct",
                "comment": "comment",
            },
        },
        "correction_evidence": {"words": {"衰草": "枯黄的草"}, "sentences": {}},
        "corrected_sentiment": {
            "primary": "惜别感伤",
            "secondary": [],
            "rationale": "some rationale",
        },
        "corrected_answer": {
            "idx": 0,
            "ans_qa_words": {"衰草": "枯黄的草"},
            "ans_qa_sents": {"故关衰草遍，离别自堪悲": "旧关一带长满枯草"},
        },
        "quality_flags": [],
    }
    stats, _ = run_test([tc], desc="emotion_error inconsistent")
    assert stats["filtered_by_rules"] == 1
    print("PASS: teacher_critique emotion_error inconsistency filtered")


def test_unexpected_choose_id_in_draft():
    s = make_valid_base()
    s["draft_answer"]["choose_id"] = "A"
    stats, _ = run_test([s], desc="unexpected choose_id in draft_answer")
    assert stats["filtered_by_rules"] == 1
    print("PASS: unexpected choose_id in draft_answer filtered")


def test_primary_qita_missing_flag():
    s = make_valid_base()
    s["sentiment"]["primary"] = "其他"
    s["sentiment"]["rationale"] = "无法归入现有标签，需要人工确认"
    s["draft_answer"]["ans_qa_words"] = {"衰草": "枯黄的草"}
    s["evidence"]["words"] = {"衰草": {"meaning": "枯黄的草", "text_clue": "故关衰草遍", "rationale": "test"}}
    s["task"]["qa_words"] = ["衰草"]
    stats, lines = run_test([s], desc="primary=其他 without needs_human_review")
    assert stats["passed"] == 1
    # Verify needs_human_review flag was auto-added
    output = json.loads(lines[0])
    assert "needs_human_review" in output.get("quality_flags", []), \
        f"Expected needs_human_review in quality_flags, got {output.get('quality_flags')}"
    print("PASS: primary=其他 auto-adds needs_human_review flag")


def test_multi_paragraph_long_reasoning():
    s = make_valid_base()
    # 300+ chars with 4 paragraphs to trigger multi-paragraph long reasoning filter
    paragraph = "这是一段非常非常非常长的推理内容。用来测试多段长推理过滤功能是否正常工作。"
    s["sentiment"]["rationale"] = "\n\n".join([paragraph] * 4)
    stats, _ = run_test([s], desc="multi-paragraph long reasoning")
    assert stats["filtered_by_rules"] == 1, f"Expected filtered=1, got {stats}"
    print("PASS: multi-paragraph long reasoning filtered")


def test_sentiment_rationale_too_long():
    s = make_valid_base()
    s["sentiment"]["rationale"] = "长" * 121  # 121 CJK chars, >120 filter
    stats, _ = run_test([s], desc="sentiment rationale >120 filter")
    assert stats["filtered_by_rules"] == 1
    print("PASS: sentiment.rationale >120 filtered")


def test_rationale_over_100_filter():
    s = make_valid_base()
    s["evidence"]["words"]["衰草"]["rationale"] = "长" * 101
    stats, _ = run_test([s], desc="rationale >100 filter")
    assert stats["filtered_by_rules"] == 1
    print("PASS: rationale >100 filtered")


if __name__ == "__main__":
    test_helpers()
    test_valid_sample_passes()
    test_invalid_record_type()
    test_missing_required_field()
    test_forbidden_cot_field()
    test_idx_mismatch()
    test_markdown_code_block()
    test_prompt_residue()
    test_coverage_missing_word_key()
    test_coverage_extra_word_key()
    test_primary_not_in_vocab()
    test_secondary_not_in_vocab_human_review()
    test_word_answer_too_long_filter()
    test_word_answer_over_50_human_review()
    test_word_answer_over_50_strict()
    test_sent_translation_too_long()
    test_appreciation_style()
    test_dedup_identical()
    test_json_parse_error()
    test_valid_teacher_critique()
    test_teacher_critique_with_choose_id()
    test_teacher_critique_emotion_inconsistent()
    test_unexpected_choose_id_in_draft()
    test_primary_qita_missing_flag()
    test_multi_paragraph_long_reasoning()
    test_sentiment_rationale_too_long()
    test_rationale_over_100_filter()
    print("\n=== ALL TESTS PASSED ===")
