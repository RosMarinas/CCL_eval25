#!/usr/bin/env python3
"""Generate teacher training data via DeepSeek-V4-Flash API.

Thin CLI wrapper around src.data.teacher.
"""
import argparse
import logging
import sys
from pathlib import Path

from src.data.teacher import (
    load_api_key,
    load_tasks,
    load_completed_idx,
    process_batch,
    print_statistics,
)

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="Generate teacher training data via API")
    parser.add_argument("--input", required=True,
                        help="Input JSON/JSONL file or directory path")
    parser.add_argument("--type", default="short-evidence",
                        choices=["short-evidence", "teacher-critique"],
                        help="Teacher data record type")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--api-key-file", default="api-key.txt",
                        help="Path to API key file (default: api-key.txt)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of samples per API request batch (1-32)")
    parser.add_argument("--candidates", default=None,
                        help="Path to candidates JSONL (teacher-critique mode)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    try:
        api_key = load_api_key(args.api_key_file)
    except (OSError, ValueError) as e:
        logger.error("Failed to read API key from %s: %s",
                     args.api_key_file, e)
        return 1

    tasks = load_tasks(args.input, args.type)
    if not tasks:
        logger.error("No tasks loaded from %s", args.input)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed_idx = set()
    if args.resume and output_path.exists():
        completed_idx = load_completed_idx(output_path)
        logger.info("Resume: %d/%d samples already completed",
                     len(completed_idx), len(tasks))

    all_records = process_batch(
        tasks=tasks, api_key=api_key, output_path=output_path,
        record_type=args.type, batch_size=args.batch_size,
        candidates_path=args.candidates, completed_idx=completed_idx,
    )
    print_statistics(all_records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
