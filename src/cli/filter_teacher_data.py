#!/usr/bin/env python3
"""Filter teacher-generated data JSONL before training.

Thin CLI wrapper around src.data.filter.TeacherDataFilter.
"""
import argparse
import sys
from pathlib import Path

from src.data.filter import TeacherDataFilter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter teacher-generated data JSONL")
    parser.add_argument("--input", "-i", required=True,
                        help="Input JSONL file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output filtered JSONL file")
    parser.add_argument("--strict", action="store_true",
                        help="Discard samples that would go to human review")
    parser.add_argument("--report", default=None,
                        help="Filtering statistics JSON path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.report:
        report_path = Path(args.report)
    else:
        report_path = output_path.with_suffix(
            output_path.suffix + ".report.json")

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    filterer = TeacherDataFilter(strict=args.strict)
    stats = filterer.process(input_path, output_path, report_path)

    print(f"Input:     {input_path}")
    print(f"Output:    {output_path}")
    print(f"Report:    {report_path}")
    print(f"Strict:    {args.strict}")
    print(f"Total:     {stats['total_samples']}")
    print(f"Parse err: {stats['json_parse_errors']}")
    print(f"Filtered:  {stats['filtered_by_rules']} "
          f"(incl. {stats.get('strict_human_review_discarded', 0)} strict)",)
    print(f"HR flags:  {stats.get('human_review_flagged_count', 0)}")
    print(f"Dedup rm:  {stats['dedup_removed']}")
    print(f"Passed:    {stats['passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
