"""
Visually inspect a prepared dataset split before training.

Usage:
    python data/inspect.py                              # 3 random from train.jsonl
    python data/inspect.py --split val                 # from val.jsonl
    python data/inspect.py --split test_held_out --n 5
    python data/inspect.py --data-dir data             # specify the data directory
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

# ── Width of the separator line ──────────────────────────────────────────────
WIDTH = 66
SEP = "═" * WIDTH  # ══════...


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def print_example(rank: int, total: int, idx: int, record: dict) -> None:
    call_type = "MULTI-CALL" if record.get("is_multi_call") else "SINGLE-CALL"
    print(SEP)
    print(f" Example {rank} / {total}  [index {idx}]  [{call_type}]")
    print(SEP)
    print(record["text"])


def print_summary(path: Path, records: list[dict]) -> None:
    total = len(records)
    multi = sum(1 for r in records if r.get("is_multi_call"))
    single = total - multi

    single_pct = (single / total * 100) if total else 0.0
    multi_pct = (multi / total * 100) if total else 0.0

    lengths = [len(r["text"]) for r in records]
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    avg_len = round(statistics.mean(lengths)) if lengths else 0

    print(SEP)
    print()
    print(f"Summary: {path}")
    print(f"  Total rows  : {total:,}")
    print(f"  Single-call : {single:,} ({single_pct:.1f}%)")
    print(f"  Multi-call  : {multi:,} ({multi_pct:.1f}%)")
    print(f"  Text length : min={min_len}  max={max_len}  avg={avg_len}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visually inspect a prepared dataset split before training."
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Which split to inspect: train | val | test_held_out (default: train).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Number of random examples to display (default: 3).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the JSONL files (default: data).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    jsonl_path = data_dir / f"{args.split}.jsonl"

    if not jsonl_path.exists():
        print(
            f"ERROR: '{jsonl_path}' not found.\n"
            "Run `make data` first to prepare the dataset.",
            file=sys.stderr,
        )
        sys.exit(1)

    records = load_jsonl(jsonl_path)
    total = len(records)

    if total == 0:
        print(f"ERROR: '{jsonl_path}' is empty.", file=sys.stderr)
        sys.exit(1)

    n = min(args.n, total)
    rng = random.Random(args.seed)
    chosen_indices = rng.sample(range(total), n)

    for rank, idx in enumerate(chosen_indices, start=1):
        print_example(rank, n, idx, records[idx])
        print()

    print_summary(jsonl_path, records)
    print()


if __name__ == "__main__":
    main()
