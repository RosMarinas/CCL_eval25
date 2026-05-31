#!/usr/bin/env python3
"""Generate synthetic perturbed candidate answers for teacher-critique.

Thin CLI wrapper around src.data.candidates.
"""
import argparse
import sys
from pathlib import Path

from src.data.candidates import PERTURBATION_TYPES, process


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic perturbed candidate answers"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input filtered short-evidence JSONL")
    parser.add_argument("--output", "-o", required=True,
                        help="Output candidates JSONL")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--compat-mode", action="store_true",
                        help="Shorten word_swap wrong meanings for tests")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    stats = process(input_path, output_path, seed=args.seed,
                    compat_mode=args.compat_mode)

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Seed:    {args.seed}")
    print(f"Compat:  {args.compat_mode}")
    print(f"Total:   {stats['total']}")
    print(f"OK:      {stats['succeeded']}")
    print(f"Failed:  {stats['failed']}")
    for ptype in PERTURBATION_TYPES:
        count = stats["type_counts"].get(ptype, 0)
        pct = count / stats["succeeded"] * 100 if stats["succeeded"] > 0 else 0.0
        print(f"  {ptype}: {count} ({pct:.1f}%)")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
