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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

DEFAULT_TEMPERATURE = 0.3
DEGRADE_TEMPERATURE = 0.5
MAX_TOKENS = 4096

DEGRADE_THRESHOLD = 3  # consecutive parse failures before degrading temperature
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
        choices=["short-evidence"],
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


def load_tasks(input_path: str) -> list[dict[str, Any]]:
    """Load tasks from a directory or a single JSON file.

    Directory input (e.g. train-data/):
        - Walks *.json files recursively (sorted by path)
        - Assumes each file is a JSON array of training samples
        - Each training sample has: title, content, keywords (object), trans, emotion
        - Constructs unified schema with sequential idx, qa_words from keywords keys,
          qa_sents from content sentence-splitting, choose = {}

    File input (e.g. eval-dev-50.json):
        - Assumes unified schema already present: idx, title, author, content,
          qa_words, qa_sents, choose
    """
    path = _resolve_path(input_path)
    if not path.exists():
        print(f"[FATAL] Input path not found: {path}", file=sys.stderr)
        sys.exit(1)
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


# ============================================================
# API call
# ============================================================


def call_teacher(
    prompt: str,
    model_name: str,
    api_key: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str | None:
    """Call the DeepSeek chat completion API and return the response text.

    Implements exponential backoff retry for 429 and 5xx errors.
    Non-retryable 4xx errors (except 429) and 401 fail immediately.

    Returns:
        Response text on success, None on failure after all retries.
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1",
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_TOKENS,
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


def build_error_record(
    task: dict[str, Any],
    model_name: str,
    error_type: str,
    error_detail: str = "",
) -> dict[str, Any]:
    """Build a placeholder record for failed generations.

    These records have record_type suffixed with '_parse_failed' or similar
    to distinguish them from successful generations.
    """
    return {
        "record_type": "short_evidence_parse_failed",
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


def process_sample(
    task: dict[str, Any],
    model_name: str,
    api_key: str,
    state: list[int],
) -> dict[str, Any]:
    """Process a single sample: build prompt, call API, parse JSON response.

    Args:
        task: Unified task dict.
        model_name: API model name.
        api_key: API key string.
        state: Mutable list with one element [consecutive_failures].
               Shared across threads for degradation logic.

    Returns:
        Output record (success or error).
    """
    prompt = build_prompt(task)

    # Degradation: if 3+ consecutive parse failures, try with higher temperature
    consecutive_failures = state[0]
    temperature = DEGRADE_TEMPERATURE if consecutive_failures >= DEGRADE_THRESHOLD else DEFAULT_TEMPERATURE

    if temperature != DEFAULT_TEMPERATURE:
        print(
            f"  idx={task['idx']}: degraded temperature={temperature} "
            f"({consecutive_failures} consecutive failures)",
            file=sys.stderr,
        )

    text = call_teacher(prompt, model_name, api_key, temperature=temperature)

    if text is None:
        # API call failed after retries
        state[0] += 1
        return build_error_record(task, model_name, "api_error", "API call failed after retries")

    parsed = extract_json(text)

    if parsed is None:
        # JSON parse failed
        state[0] += 1
        return build_error_record(
            task,
            model_name,
            "parse_failed",
            f"Could not parse JSON from response (length={len(text)})",
        )

    # Validate required fields
    required = ["evidence", "sentiment", "draft_answer"]
    missing = [f for f in required if f not in parsed or not isinstance(parsed[f], dict)]
    if missing:
        state[0] += 1
        return build_error_record(
            task,
            model_name,
            "missing_fields",
            f"Missing required fields: {', '.join(missing)}",
        )

    # Success: reset consecutive failure counter
    state[0] = 0

    return build_output_record(task, parsed, model_name)


# ============================================================
# Batch processing
# ============================================================


def process_batch(
    tasks: list[dict[str, Any]],
    model_name: str,
    api_key: str,
    rate_limit: int,
) -> list[dict[str, Any]]:
    """Process a batch of tasks concurrently with rate-limited parallelism.

    Each task is submitted to a thread pool (max_workers=rate_limit).
    Results are collected and sorted by idx for a deterministic output order.
    """
    records: list[dict[str, Any]] = []
    # Shared mutable state: [consecutive_failures]
    # Using a list so threads can read/write the counter.
    state = [0]

    with ThreadPoolExecutor(max_workers=rate_limit) as executor:
        future_to_idx = {
            executor.submit(process_sample, task, model_name, api_key, state): task["idx"]
            for task in tasks
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                record = future.result()
                records.append(record)
                record_type = record.get("record_type", "")
                if record_type == "short_evidence":
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
    success = sum(1 for r in all_records if r.get("record_type") == "short_evidence")
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
    3. Load tasks from input (directory or file)
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
    all_tasks = load_tasks(args.input)
    print(f"[INFO] Total tasks loaded: {len(all_tasks)}")

    if not all_tasks:
        print("[WARN] No tasks to process.", file=sys.stderr)
        return 0

    # ---- Resume: skip already-completed idx ----
    completed_idx = load_completed_idx(output_path)
    if completed_idx:
        print(f"[INFO] Found {len(completed_idx)} already completed idx in output file")

    pending = [t for t in all_tasks if t["idx"] not in completed_idx]
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
        first_idx = batch[0]["idx"]
        last_idx = batch[-1]["idx"]

        print(
            f"\n[Batch] idx range [{first_idx}..{last_idx}] "
            f"({batch_start + 1}-{batch_end_idx} of {total_pending})"
        )

        records = process_batch(batch, args.model, api_key, args.rate_limit)

        # Write immediately for resume/crash safety
        append_records(output_path, records)
        all_records.extend(records)

        # Brief pause between batches
        if batch_start + args.batch_size < total_pending:
            time.sleep(0.5)

    # ---- Print final statistics ----
    print_statistics(all_records)

    has_failures = any(
        r.get("record_type") != "short_evidence" for r in all_records
    )
    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
