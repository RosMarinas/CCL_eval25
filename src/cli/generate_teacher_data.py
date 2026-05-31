#!/usr/bin/env python3
"""Generate teacher data for CCL25 classical Chinese poetry evaluation.

Two-stage pipeline:
  Stage 1 (Reasoner): Outputs evidence + sentiment + draft_answer.
                       Does NOT output choose_id.
  Stage 2 (Formatter): Maps sentiment -> choose_id (separate script).

This script handles Stage 1 generation via the DeepSeek-V4-Flash API.
Output is a JSONL file with one complete teacher-data record per line.

Usage:
    python src/cli/generate_teacher_data.py \
        --input data/train-data \
        --type short-evidence \
        --model deepseek-v4-flash \
        --api-key-file api-key.txt \
        --output data/teacher/train-short-evidence.jsonl \
        --batch-size 10 \
        --rate-limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any

from openai import OpenAI

# ============================================================
# Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONTROLLED_VOCABULARY = [
    # 离别
    "惜别感伤",
    "送别不舍",
    "离别愁绪",
    # 思乡
    "思乡怀远",
    "羁旅思归",
    "故园之思",
    # 忧国
    "忧国伤时",
    "报国壮志",
    "兴亡之叹",
    # 山水田园
    "山水闲适",
    "田园之乐",
    "隐逸情怀",
    # 怀古
    "怀古伤今",
    "历史沧桑",
    "昔盛今衰",
    # 爱情闺怨
    "相思闺怨",
    "爱情甜蜜",
    "相思之苦",
    # 人生感慨
    "人生无常",
    "时光易逝",
    "仕途失意",
    # 边塞战争
    "边塞征战",
    "将士艰辛",
    "厌战思归",
    # 兜底
    "其他",
]

SENTENCE_DELIMITERS = "。！？!?"

MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0

MAX_TOKENS = 4096
THINKING_MODE = {"type": "enabled"}
DEFAULT_REASONING_EFFORT = "high"
DEGRADE_REASONING_EFFORT = "max"

DEGRADE_THRESHOLD = 3  # consecutive parse failures before increasing reasoning effort
DEGRADE_RETRY_FAILURE = "parse_failed"

# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate teacher data for CCL25 classical Chinese poetry evaluation. "
            "Uses DeepSeek-V4-Flash API to produce structured evidence, sentiment "
            "analysis, and draft answers for training a Qwen3-8B model via QLoRA."
        ),
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Input path: directory (e.g. data/train-data/) with train.json files, "
            "or a JSON file (e.g. data/eval_data.json) in unified schema format"
        ),
    )
    parser.add_argument(
        "--type",
        type=str,
        default="short-evidence",
        choices=["short-evidence", "teacher-critique"],
        help="Record type to generate (default: short-evidence)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-v4-flash",
        help="Teacher model name passed to the API (default: deepseek-v4-flash)",
    )
    parser.add_argument(
        "--api-key-file",
        type=str,
        default="api-key.txt",
        help="Path to API key file, relative to project root (default: api-key.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path (absolute or relative to project root)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of samples per batch (default: 10)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=5,
        help="Max concurrent API requests (default: 5)",
    )
    return parser.parse_args()


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
# API key loading
# ============================================================


def load_api_key(api_key_path: str) -> str:
    """Load API key from file. NEVER print the key to logs or error messages."""
    path = _resolve_path(api_key_path)
    if not path.exists():
        print(f"[FATAL] API key file not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(path, "r") as f:
            key = f.readline().strip()
        if not key:
            print("[FATAL] API key file is empty", file=sys.stderr)
            sys.exit(1)
        return key
    except OSError as e:
        print(f"[FATAL] Cannot read API key file: {e}", file=sys.stderr)
        sys.exit(1)


# ============================================================
# Task loading
# ============================================================


def load_tasks(input_path: str, record_type: str = "short-evidence") -> list[dict[str, Any]]:
    """Load tasks (and optionally candidate answers) from input.

    For short-evidence:
        Directory input (e.g. train-data/):
            - Walks *.json files recursively (sorted by path)
            - Assumes each file is a JSON array of training samples
            - Each training sample has: title, content, keywords (object), trans, emotion
            - Constructs unified schema with sequential idx, qa_words from keywords keys,
              qa_sents from content sentence-splitting, choose = {}
        File input (e.g. eval-dev-50.json):
            - Assumes unified schema already present: idx, title, author, content,
              qa_words, qa_sents, choose

    For teacher-critique:
        Input must be a JSONL file where each line has:
            {"task": {...}, "candidate": {...}}
        Returns list of dicts with "task" and "candidate" keys.
    """
    path = _resolve_path(input_path)
    if not path.exists():
        print(f"[FATAL] Input path not found: {path}", file=sys.stderr)
        sys.exit(1)
    if record_type == "teacher-critique":
        return _load_tasks_from_jsonl(path)
    if path.is_dir():
        return _load_tasks_from_dir(path)
    return _load_tasks_from_file(path)


def _load_tasks_from_dir(directory: Path) -> list[dict[str, Any]]:
    """Load training samples from a directory tree of train.json files.

    Each train.json is a JSON array of samples with:
        title, content, keywords (object), trans, emotion

    The unified schema fields are constructed as:
        idx: sequential (0, 1, 2, ...)
        qa_words: all keys from the keywords object
        qa_sents: content split by Chinese sentence-ending punctuation
        choose: {} (empty -- training data has no options)
    """
    tasks: list[dict[str, Any]] = []
    idx = 0
    for json_path in sorted(directory.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[WARN] Skipping unreadable file {json_path}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"[WARN] Expected JSON array in {json_path}, got {type(data).__name__}", file=sys.stderr)
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            content = item.get("content", "") or ""
            task = {
                "idx": idx,
                "title": item.get("title", "") or "",
                "author": item.get("author", "") or "",
                "content": content,
                "qa_words": list(item.get("keywords", {}).keys()),
                "qa_sents": _split_sentences(content),
                "choose": {},
            }
            tasks.append(task)
            idx += 1
    if not tasks:
        print(f"[WARN] No tasks found in directory: {directory}", file=sys.stderr)
    else:
        print(f"  Loaded {len(tasks)} training tasks from {directory}")
    return tasks


def _load_tasks_from_file(path: Path) -> list[dict[str, Any]]:
    """Load tasks from a single JSON file in unified schema format."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[FATAL] Cannot read input file {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if isinstance(data, dict):
        data = [data]

    for task in data:
        # Ensure all unified schema fields are present
        task.setdefault("title", "")
        task.setdefault("author", "")
        task.setdefault("choose", {})
        task.setdefault("qa_words", [])
        task.setdefault("qa_sents", [])
        # Ensure idx is present
        if "idx" not in task:
            print(f"[WARN] Task missing 'idx' field; will be lost", file=sys.stderr)

    print(f"  Loaded {len(data)} eval tasks from {path}")
    return data


def _load_tasks_from_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load task-candidate pairs from a JSONL file for teacher-critique.

    Each line should be a JSON object with keys:
        task: unified task dict
        candidate: candidate answer dict (perturbed/incorrect answer)
    """
    items: list[dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_num}: JSON decode error: {e}", file=sys.stderr)
                    continue
                if not isinstance(obj, dict):
                    print(f"[WARN] Line {line_num}: Expected JSON object, got {type(obj).__name__}", file=sys.stderr)
                    continue
                task = obj.get("task")
                candidate = obj.get("candidate")
                if not isinstance(task, dict) or not isinstance(candidate, dict):
                    print(f"[WARN] Line {line_num}: Missing 'task' or 'candidate' key", file=sys.stderr)
                    continue
                if "idx" not in task:
                    print(f"[WARN] Line {line_num}: Task missing 'idx' field", file=sys.stderr)
                items.append({"task": task, "candidate": candidate})
    except OSError as e:
        print(f"[FATAL] Cannot read input file {path}: {e}", file=sys.stderr)
        sys.exit(1)

    if not items:
        print(f"[FATAL] No valid task-candidate pairs found in {path}", file=sys.stderr)
        sys.exit(1)

    print(f"  Loaded {len(items)} task-candidate pairs from {path}")
    return items


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


# ============================================================
# Prompt building
# ============================================================


def build_prompt(task: dict[str, Any]) -> str:
    """Build the teacher prompt for short-evidence generation.

    Uses the exact prompt template from docs/contracts/teacher-data.md Section 2.1.
    The conditional final_answer section is included only when task has
    non-empty choose options.
    """
    task_json = json.dumps(task, ensure_ascii=False, indent=2)
    has_choose = bool(task.get("choose"))

    if has_choose:
        final_answer_section = (
            '\n  ,"final_answer": {\n'
            '    "ans_qa_words": {},\n'
            '    "ans_qa_sents": {},\n'
            '    "choose_id": "<A|B|C|D>"\n'
            "  }"
        )
    else:
        final_answer_section = ""

    prompt = (
        "你是古诗词理解任务的教师模型。请根据输入题目生成结构化短证据、情感分析和草稿答案。\n"
        "\n"
        "硬性要求：\n"
        "1. 只输出一个合法 JSON 对象，不输出 Markdown、解释文字或代码块。\n"
        "2. 不要输出自由长 CoT，不要写逐步推理过程。\n"
        "3. evidence 只能写短证据：词义线索、句意骨架、情感线索；每条 rationale 不超过 40 个中文字符。\n"
        "4. ans_qa_words 必须覆盖所有 qa_words 去重后的词语，key 必须与输入完全一致。\n"
        "5. ans_qa_sents 必须覆盖所有 qa_sents 去重后的句子，key 必须与输入完全一致。\n"
        "6. 情感分析使用 `sentiment` 字段，包含 `primary`（主要情感）、`secondary`（次要情感列表，可选）"
        "和 `rationale`（判断依据）。`sentiment.primary` 和 `sentiment.secondary` 中的每个标签必须从受控词汇表中选择"
        "（惜别感伤、送别不舍、离别愁绪、思乡怀远、羁旅思归、故园之思、忧国伤时、报国壮志、兴亡之叹、山水闲适、"
        "田园之乐、隐逸情怀、怀古伤今、历史沧桑、昔盛今衰、相思闺怨、爱情甜蜜、相思之苦、人生无常、时光易逝、"
        "仕途失意、边塞征战、将士艰辛、厌战思归、其他）。若无法归入以上标签，使用\"其他\"并在 quality_flags 中标记"
        " needs_human_review。\n"
        "7. evidence.emotion 是短证据字符串数组（每条 ≤ 60 字），包含情感判断依据，不包含 option_id。\n"
        "8. draft_answer 仅包含词义和句译，不包含 choose_id。\n"
        "9. final_answer 仅在题目包含 choose 选项（非空）时才输出 choose_id，"
        "且 choose_id 必须来自 choose 的选项 ID；若 choose 为空，不输出 choose_id 字段。\n"
        '10. 若题目缺少情感选项，在 quality_flags 中加入 "missing_emotion_options"。\n'
        "\n"
        f"输入题目：\n{task_json}\n"
        "\n"
        "请输出：\n"
        "{\n"
        '  "record_type": "short_evidence",\n'
        f'  "idx": {task["idx"]},\n'
        '  "evidence": {\n'
        '    "words": {\n'
        '      "<target_word>": {\n'
        '        "meaning": "<简明词义>",\n'
        '        "text_clue": "<诗中依据，短语或句子片段>",\n'
        '        "rationale": "<短理由，不超过40字>"\n'
        "      }\n"
        "    },\n"
        '    "sentences": {\n'
        '      "<target_sentence>": {\n'
        '        "translation": "<现代汉语直译或意译>",\n'
        '        "key_images": ["<意象1>", "<意象2>"],\n'
        '        "rationale": "<短理由，不超过40字>"\n'
        "      }\n"
        "    },\n"
        '    "emotion": [\n'
        '      "<情感判断依据1，不超过60字>",\n'
        '      "<情感判断依据2，不超过60字>"\n'
        "    ]\n"
        "  },\n"
        '  "sentiment": {\n'
        '    "primary": "<主要情感标签，从受控词汇表选择>",\n'
        '    "secondary": ["<次要情感标签1>", "<次要情感标签2>"],\n'
        '    "rationale": "<情感判断依据，不超过80字，引用诗中关键词和意象>"\n'
        "  },\n"
        '  "draft_answer": {\n'
        '    "ans_qa_words": {},\n'
        '    "ans_qa_sents": {}\n'
        f"  }}{final_answer_section}\n"
        '  ,\n'
        '  "quality_flags": []\n'
        "}"
    )
    return prompt


def build_teacher_critique_prompt(task: dict[str, Any], candidate_answer: dict[str, Any]) -> str:
    """Build the teacher prompt for teacher-critique generation.

    Uses the exact prompt template from docs/contracts/teacher-data.md Section 2.2.
    Generates structured critique and corrected answer for a candidate answer.
    """
    task_json = json.dumps(task, ensure_ascii=False, indent=2)
    candidate_json = json.dumps(candidate_answer, ensure_ascii=False, indent=2)

    vocab_str = "、".join(CONTROLLED_VOCABULARY)

    prompt = (
        "你是古诗词理解任务的教师模型。请根据输入题目和一个候选错误答案，生成结构化批改意见与修正答案。\n"
        "\n"
        "硬性要求：\n"
        "1. 只输出一个合法 JSON 对象，不输出 Markdown、解释文字或代码块。\n"
        "2. 不要输出自由长 CoT，不要写逐步推理过程。\n"
        "3. critique 只能指出可验证的错误类型和短理由；每条 comment 不超过 50 个中文字符。\n"
        "4. corrected_answer 必须使用最终答案 schema。\n"
        "5. 不要改写正确且足够简洁的字段。\n"
        '6. 如果无法判断某字段是否错误，将 error_type 写为 "uncertain"，并在 quality_flags 中加入 "needs_human_review"。\n'
        "7. emotion_error 评价候选答案中 sentiment 分析的准确性，而非 choose_id 的正误。\n"
        "8. sentiment.primary 和 sentiment.secondary 中的每个标签必须来自受控词汇表"
        f"（{vocab_str}）。\n"
        "\n"
        f"输入题目：\n{task_json}\n"
        "\n"
        f"候选答案：\n{candidate_json}\n"
        "\n"
        "请输出：\n"
        "{\n"
        '  "record_type": "teacher_critique",\n'
        f'  "idx": {task["idx"]},\n'
        f'  "candidate_answer": {candidate_json},\n'
        '  "critique": {\n'
        '    "word_errors": [\n'
        "      {\n"
        '        "target": "<target_word>",\n'
        '        "error_type": "<missing|wrong_meaning|overlong|unsupported|correct|uncertain>",\n'
        '        "comment": "<短批注意见>"\n'
        "      }\n"
        "    ],\n"
        '    "sentence_errors": [\n'
        "      {\n"
        '        "target": "<target_sentence>",\n'
        '        "error_type": "<missing|wrong_translation|overlong|unsupported|correct|uncertain>",\n'
        '        "comment": "<短批注意见>"\n'
        "      }\n"
        "    ],\n"
        '    "emotion_error": {\n'
        '      "candidate_primary": "<候选答案中的 sentiment.primary>",\n'
        '      "correct_primary": "<正确的 sentiment.primary 标签>",\n'
        '      "candidate_secondary": ["<候选答案中的 sentiment.secondary 标签列表>"],\n'
        '      "correct_secondary": ["<正确的 sentiment.secondary 标签列表>"],\n'
        '      "primary_error_type": "<wrong_label|not_in_vocab|missing|correct|uncertain>",\n'
        '      "secondary_error_type": "<extra_label|missing_label|wrong_label|correct|uncertain>",\n'
        '      "rationale_error_type": "<no_evidence|contradicts_vocab|correct|uncertain>",\n'
        '      "comment": "<短批注意见>"\n'
        "    }\n"
        "  },\n"
        '  "correction_evidence": {\n'
        '    "words": {},\n'
        '    "sentences": {},\n'
        '    "emotion": [\n'
        '      "<情感判断依据1>",\n'
        '      "<情感判断依据2>"\n'
        "    ]\n"
        "  },\n"
        '  "corrected_sentiment": {\n'
        '    "primary": "<正确的 sentiment.primary 标签>",\n'
        '    "secondary": ["<正确的 sentiment.secondary 标签列表>"],\n'
        '    "rationale": "<正确的情感判断依据>"\n'
        "  },\n"
        '  "corrected_answer": {\n'
        f'    "idx": {task["idx"]},\n'
        '    "ans_qa_words": {},\n'
        '    "ans_qa_sents": {}\n'
        "  },\n"
        '  "quality_flags": []\n'
        "}"
    )
    return prompt


# ============================================================
# API call
# ============================================================


def call_teacher(
    prompt: str,
    model_name: str,
    api_key: str,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> str | None:
    """Call the DeepSeek chat completion API and return the response text.

    Implements exponential backoff retry for 429 and 5xx errors.
    Non-retryable 4xx errors (except 429) and 401 fail immediately.

    Returns:
        Response text on success, None on failure after all retries.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
                reasoning_effort=reasoning_effort,
                extra_body={"thinking": THINKING_MODE},
            )
            content = response.choices[0].message.content
            return content

        except Exception as exc:
            status_code = _extract_status_code(exc)

            # Non-retryable errors: 4xx except 429
            if status_code is not None and 400 <= status_code < 500 and status_code != 429:
                if status_code == 401:
                    print("[ERROR] Authentication failed (401). Check API key.", file=sys.stderr)
                else:
                    print(f"[ERROR] Non-retryable HTTP {status_code}", file=sys.stderr)
                return None

            # Retryable: 429 or 5xx
            if attempt < MAX_RETRIES:
                delay = (2**attempt) + random.random()
                print(
                    f"[RETRY] Attempt {attempt + 1}/{MAX_RETRIES} failed: {exc}. "
                    f"Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"[RETRY_EXHAUSTED] Max retries ({MAX_RETRIES}) reached: {exc}",
                    file=sys.stderr,
                )
                return None

    return None


def _extract_status_code(exc: Exception) -> int | None:
    """Try to extract HTTP status code from an API error exception.

    Handles openai.APIStatusError (status_code), openai.APIError (code),
    and other common patterns.
    """
    # openai.APIStatusError and similar
    if hasattr(exc, "status_code"):
        try:
            return int(getattr(exc, "status_code"))
        except (ValueError, TypeError):
            pass
    # openai.APIError
    if hasattr(exc, "code"):
        try:
            return int(getattr(exc, "code"))
        except (ValueError, TypeError):
            pass
    # httpx status_code
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):  # type: ignore[union-attr]
        try:
            return int(exc.response.status_code)  # type: ignore[union-attr]
        except (ValueError, TypeError):
            pass
    return None


# ============================================================
# Response parsing
# ============================================================


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from model response text.

    Handles three scenarios:
    1. Pure JSON string (direct parse)
    2. Markdown code block (```json ... ```)
    3. JSON embedded in text ({...} regex extraction)
    """
    text = text.strip()
    if not text:
        return None

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Markdown code block: ```json ... ```
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fallback: extract first { ... } block
    brace_depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
            if brace_depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    start = -1  # reset, try next top-level block
    return None


# ============================================================
# Output record construction
# ============================================================


def build_output_record(
    task: dict[str, Any],
    parsed: dict[str, Any],
    model_name: str,
    prompt_version: str = "teacher-data-v2",
) -> dict[str, Any]:
    """Build the complete output record with source and task metadata.

    Wraps the model's parsed JSON with additional fields (source, task).
    Includes final_answer only when the task has non-empty choose options.
    """
    # Copy fields from model output, with empty fallbacks for robustness
    evidence = parsed.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    sentiment = parsed.get("sentiment")
    if not isinstance(sentiment, dict):
        sentiment = {}

    draft_answer = parsed.get("draft_answer")
    if not isinstance(draft_answer, dict):
        draft_answer = {}

    quality_flags = parsed.get("quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []

    record: dict[str, Any] = {
        "record_type": "short_evidence",
        "idx": task["idx"],
        "source": {
            "teacher_model": model_name,
            "prompt_version": prompt_version,
            "created_at": date.today().strftime("%Y-%m-%d"),
        },
        "task": task,
        "evidence": evidence,
        "sentiment": sentiment,
        "draft_answer": draft_answer,
        "quality_flags": quality_flags,
    }

    # Add final_answer if task has choose options
    if task.get("choose"):
        fa = parsed.get("final_answer")
        if not isinstance(fa, dict):
            fa = {}
        record["final_answer"] = {
            "idx": task["idx"],
            "ans_qa_words": fa.get(
                "ans_qa_words",
                draft_answer.get("ans_qa_words", {}),
            ),
            "ans_qa_sents": fa.get(
                "ans_qa_sents",
                draft_answer.get("ans_qa_sents", {}),
            ),
            "choose_id": fa.get("choose_id", ""),
        }

    return record


def build_teacher_critique_output_record(
    task: dict[str, Any],
    parsed: dict[str, Any],
    model_name: str,
    candidate_answer: dict[str, Any],
    prompt_version: str = "teacher-critique-v2",
) -> dict[str, Any]:
    """Build the complete output record for teacher-critique with source and task metadata.

    Wraps the model's parsed JSON with additional fields (source, task).
    Follows the Section 4 schema from docs/contracts/teacher-data.md.
    Uses the input candidate_answer directly (not the model's reproduced version).
    """
    critique = parsed.get("critique")
    if not isinstance(critique, dict):
        critique = {}

    correction_evidence = parsed.get("correction_evidence")
    if not isinstance(correction_evidence, dict):
        correction_evidence = {}

    corrected_sentiment = parsed.get("corrected_sentiment")
    if not isinstance(corrected_sentiment, dict):
        corrected_sentiment = {}

    corrected_answer = parsed.get("corrected_answer")
    if not isinstance(corrected_answer, dict):
        corrected_answer = {}

    quality_flags = parsed.get("quality_flags")
    if not isinstance(quality_flags, list):
        quality_flags = []

    record: dict[str, Any] = {
        "record_type": "teacher_critique",
        "idx": task["idx"],
        "source": {
            "teacher_model": model_name,
            "prompt_version": prompt_version,
            "created_at": date.today().strftime("%Y-%m-%d"),
            "candidate_source": "synthetic",
        },
        "task": task,
        "candidate_answer": candidate_answer,
        "critique": critique,
        "correction_evidence": correction_evidence,
        "corrected_sentiment": corrected_sentiment,
        "corrected_answer": corrected_answer,
        "quality_flags": quality_flags,
    }

    return record


def build_error_record(
    task: dict[str, Any],
    model_name: str,
    error_type: str,
    error_detail: str = "",
    record_type_str: str = "short_evidence",
) -> dict[str, Any]:
    """Build a placeholder record for failed generations.

    These records have record_type suffixed with '_<error_type>'
    to distinguish them from successful generations.
    """
    return {
        "record_type": f"{record_type_str}_{error_type}",
        "idx": task["idx"],
        "source": {
            "teacher_model": model_name,
            "prompt_version": "teacher-data-v2",
            "created_at": date.today().strftime("%Y-%m-%d"),
        },
        "task": task,
        "error_type": error_type,
        "error_detail": error_detail,
    }


# ============================================================
# Sample processing
# ============================================================


def _get_item_idx(item: dict[str, Any], record_type: str) -> int:
    """Extract the integer idx from a task item, regardless of record type."""
    if record_type == "teacher-critique":
        return item["task"]["idx"]
    return item["idx"]


def process_sample(
    item: dict[str, Any],
    model_name: str,
    api_key: str,
    state: list[int],
    record_type: str = "short-evidence",
) -> dict[str, Any]:
    """Process a single sample: build prompt, call API, parse JSON response.

    Args:
        item: Unified task dict (short-evidence) or {"task": ..., "candidate": ...} (teacher-critique).
        model_name: API model name.
        api_key: API key string.
        state: Mutable list with one element [consecutive_failures].
               Shared across threads for degradation logic.
        record_type: "short-evidence" or "teacher-critique".

    Returns:
        Output record (success or error).
    """
    record_type_underscore = record_type.replace("-", "_")

    if record_type == "teacher-critique":
        task = item["task"]
        candidate_answer = item["candidate"]
        prompt = build_teacher_critique_prompt(task, candidate_answer)
        required_fields = ["critique", "correction_evidence", "corrected_sentiment", "corrected_answer"]
    else:
        task = item
        candidate_answer = None
        prompt = build_prompt(task)
        required_fields = ["evidence", "sentiment", "draft_answer"]

    # Degradation: thinking mode ignores temperature, so raise reasoning effort instead.
    consecutive_failures = state[0]
    reasoning_effort = (
        DEGRADE_REASONING_EFFORT
        if consecutive_failures >= DEGRADE_THRESHOLD
        else DEFAULT_REASONING_EFFORT
    )

    if reasoning_effort != DEFAULT_REASONING_EFFORT:
        print(
            f"  idx={_get_item_idx(item, record_type)}: degraded reasoning_effort={reasoning_effort} "
            f"({consecutive_failures} consecutive failures)",
            file=sys.stderr,
        )

    text = call_teacher(prompt, model_name, api_key, reasoning_effort=reasoning_effort)

    if text is None:
        # API call failed after retries
        state[0] += 1
        return build_error_record(task, model_name, "api_error", "API call failed after retries", record_type_str=record_type_underscore)

    parsed = extract_json(text)

    if parsed is None:
        # JSON parse failed
        state[0] += 1
        return build_error_record(
            task,
            model_name,
            "parse_failed",
            f"Could not parse JSON from response (length={len(text)})",
            record_type_str=record_type_underscore,
        )

    # Validate required fields
    missing = [f for f in required_fields if f not in parsed or not isinstance(parsed[f], dict)]
    if missing:
        state[0] += 1
        return build_error_record(
            task,
            model_name,
            "missing_fields",
            f"Missing required fields: {', '.join(missing)}",
            record_type_str=record_type_underscore,
        )

    # Success: reset consecutive failure counter
    state[0] = 0

    if record_type == "teacher-critique":
        return build_teacher_critique_output_record(task, parsed, model_name, candidate_answer)
    return build_output_record(task, parsed, model_name)


# ============================================================
# Batch processing
# ============================================================


def process_batch(
    items: list[dict[str, Any]],
    model_name: str,
    api_key: str,
    rate_limit: int,
    record_type: str = "short-evidence",
) -> list[dict[str, Any]]:
    """Process a batch of items concurrently with rate-limited parallelism.

    Each item is submitted to a thread pool (max_workers=rate_limit).
    Results are collected and sorted by idx for a deterministic output order.
    """
    records: list[dict[str, Any]] = []
    # Shared mutable state: [consecutive_failures]
    # Using a list so threads can read/write the counter.
    state = [0]

    success_record_types = ("short_evidence", "teacher_critique")
    record_type_underscore = record_type.replace("-", "_")

    with ThreadPoolExecutor(max_workers=rate_limit) as executor:
        future_to_idx = {
            executor.submit(process_sample, item, model_name, api_key, state, record_type): _get_item_idx(item, record_type)
            for item in items
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                record = future.result()
                records.append(record)
                rec_type = record.get("record_type", "")
                if rec_type in success_record_types:
                    print(f"  idx={idx}: OK")
                else:
                    error_type = record.get("error_type", "?")
                    print(f"  idx={idx}: FAIL({error_type})")
            except Exception as exc:
                print(f"  idx={idx}: UNEXPECTED: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                records.append(
                    build_error_record(
                        {"idx": idx, "title": "", "author": "", "content": "", "qa_words": [], "qa_sents": [], "choose": {}},
                        model_name,
                        "unexpected_error",
                        str(exc),
                        record_type_str=record_type_underscore,
                    )
                )

    # Sort by idx for deterministic output order
    records.sort(key=lambda r: r.get("idx", -1))
    return records


# ============================================================
# Resume support
# ============================================================


def load_completed_idx(output_path: Path) -> set[int]:
    """Load idx values already present in the output JSONL file.

    Reads each line, extracts the idx field, and returns the set.
    Used for resume support: skips samples that were already processed.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    completed: set[int] = set()
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    idx = record.get("idx")
                    if idx is not None:
                        completed.add(idx)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return completed


def append_records(output_path: Path, records: list[dict[str, Any]]) -> None:
    """Append records to the output JSONL file (one JSON object per line).

    Creates parent directories if they do not exist.
    Opens in append mode so multiple batch writes accumulate correctly.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Statistics
# ============================================================


def print_statistics(all_records: list[dict[str, Any]]) -> None:
    """Print summary statistics after all batches complete."""
    total = len(all_records)
    success = sum(1 for r in all_records if r.get("record_type") in ("short_evidence", "teacher_critique"))
    failed_api = sum(1 for r in all_records if r.get("error_type") == "api_error")
    failed_parse = sum(1 for r in all_records if r.get("error_type") == "parse_failed")
    failed_missing = sum(1 for r in all_records if r.get("error_type") == "missing_fields")
    failed_unexpected = sum(1 for r in all_records if r.get("error_type") == "unexpected_error")

    success_rate = (success / total * 100) if total > 0 else 0.0

    print(f"\n{'=' * 50}")
    print("Generation Complete")
    print(f"{'=' * 50}")
    print(f"  Total samples:      {total}")
    print(f"  Success:            {success} ({success_rate:.1f}%)")
    print(f"  API errors:         {failed_api}")
    print(f"  Parse errors:       {failed_parse}")
    print(f"  Missing fields:     {failed_missing}")
    print(f"  Unexpected errors:  {failed_unexpected}")
    print(f"{'=' * 50}")


# ============================================================
# Main entry point
# ============================================================


def main() -> int:
    """Execute the teacher data generation pipeline.

    1. Parse CLI arguments
    2. Load API key
    3. Load tasks from input (directory, file, or JSONL)
    4. Resume: skip already-processed idx from output file
    5. Process pending tasks in batches
    6. Write results incrementally
    7. Print summary statistics

    Returns:
        0 if all samples succeeded, 1 if any failures occurred.
    """
    args = parse_args()

    # ---- Resolve paths ----
    output_path = _resolve_path(args.output)

    # ---- Load API key (never print the key) ----
    api_key = load_api_key(args.api_key_file)

    # ---- Load tasks ----
    print(f"[INFO] Loading tasks from: {args.input}")
    all_tasks = load_tasks(args.input, args.type)
    print(f"[INFO] Total tasks loaded: {len(all_tasks)}")

    if not all_tasks:
        print("[WARN] No tasks to process.", file=sys.stderr)
        return 0

    # ---- Resume: skip already-completed idx ----
    completed_idx = load_completed_idx(output_path)
    if completed_idx:
        print(f"[INFO] Found {len(completed_idx)} already completed idx in output file")

    success_record_types = ("short_evidence", "teacher_critique")
    pending = [t for t in all_tasks if _get_item_idx(t, args.type) not in completed_idx]
    if not pending:
        print("[INFO] All tasks already completed. Nothing to do.")
        return 0

    print(f"[INFO] Pending tasks: {len(pending)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Batch size: {args.batch_size}, Rate limit: {args.rate_limit}")
    print(f"[INFO] Output: {output_path}")

    # ---- Process batches ----
    all_records: list[dict[str, Any]] = []
    total_pending = len(pending)

    for batch_start in range(0, total_pending, args.batch_size):
        batch = pending[batch_start : batch_start + args.batch_size]
        batch_end_idx = min(batch_start + args.batch_size, total_pending)
        first_idx = _get_item_idx(batch[0], args.type)
        last_idx = _get_item_idx(batch[-1], args.type)

        print(
            f"\n[Batch] idx range [{first_idx}..{last_idx}] "
            f"({batch_start + 1}-{batch_end_idx} of {total_pending})"
        )

        records = process_batch(batch, args.model, api_key, args.rate_limit, args.type)

        # Write immediately for resume/crash safety
        append_records(output_path, records)
        all_records.extend(records)

        # Brief pause between batches
        if batch_start + args.batch_size < total_pending:
            time.sleep(0.5)

    # ---- Print final statistics ----
    print_statistics(all_records)

    has_failures = any(
        r.get("record_type") not in success_record_types for r in all_records
    )
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
