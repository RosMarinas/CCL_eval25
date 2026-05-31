#!/usr/bin/env python3
"""Build training datasets for B8/BC8 QLoRA training.

Thin CLI wrapper around src.data.builder.
"""
import argparse
import sys

from src.data.builder import (
    RANDOM_SEED,
    build_answer_only,
    build_bc8_mixed,
    build_sentiment_mapping,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build training datasets for B8/BC8 QLoRA training.",
    )
    subparsers = parser.add_subparsers(dest="type", required=True)

    ao = subparsers.add_parser("answer-only")
    ao.add_argument("--keywords", required=True,
                     help="Path to train-data/ directory")
    ao.add_argument("--teacher", required=True,
                     help="Path to teacher data JSONL")
    ao.add_argument("--output", "-o", required=True,
                     help="Output JSONL file path")

    bc8 = subparsers.add_parser("bc8-mixed")
    bc8.add_argument("--ratio", default="50-25-25")
    bc8.add_argument("--answer-only", required=True)
    bc8.add_argument("--short-evidence", required=True)
    bc8.add_argument("--teacher-critique", default=None)
    bc8.add_argument("--output-dir", required=True)
    bc8.add_argument("--seed", type=int, default=RANDOM_SEED)

    sm = subparsers.add_parser("sentiment-mapping")
    sm.add_argument("--teacher", required=True)
    sm.add_argument("--output", "-o", required=True)

    args = parser.parse_args()

    print(f"CCL25 Training Data Builder (type={args.type})")
    print(f"{'=' * 50}")

    if args.type == "answer-only":
        return build_answer_only(
            keywords=args.keywords, teacher=args.teacher, output=args.output)
    elif args.type == "bc8-mixed":
        return build_bc8_mixed(
            answer_only=getattr(args, "answer-only"),
            short_evidence=getattr(args, "short-evidence"),
            teacher_critique=args.teacher_critique,
            ratio=args.ratio, output_dir=args.output_dir, seed=args.seed)
    elif args.type == "sentiment-mapping":
        return build_sentiment_mapping(
            teacher=args.teacher, output=args.output)
    else:
        print(f"[FATAL] Unknown type: {args.type}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
