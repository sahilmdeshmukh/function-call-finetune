"""
Prepares Salesforce/xlam-function-calling-60k for SFT on google/gemma-4-e4b-it.

Steps:
1. Load dataset from HF Hub
2. Filter rows with malformed JSON in 'tools' or 'answers'
3. Convert to Gemma message format (tool schemas folded into user turn)
4. Apply Gemma 4 tokenizer chat template to get the final training string
5. Stratify by single-call vs multi-call (multi-call = len(answers) > 1)
6. Carve out 500 examples as held-out test set (stratified, NEVER used in training)
7. Split remainder 90/10 into train/val
8. Hash-verify zero overlap between train, val, and test_held_out
9. Save to data/train.jsonl, data/val.jsonl, data/test_held_out.jsonl
10. Print summary stats
"""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Optional: load .env if python-dotenv is installed
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables directly


# ---------------------------------------------------------------------------
# Gemma formatting helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = (
    "You have access to the following functions. Use them if required:\n\n"
    "{tools_json}\n\n"
    'To call a function, respond with a JSON object of the format:\n'
    '{{"name": "function_name", "arguments": {{"key": "value"}}}}\n'
    "For multiple calls, respond with a JSON array of such objects."
)


def build_system_content(tools_list: list) -> str:
    """Format the tool schemas into the system/user preamble."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        tools_json=json.dumps(tools_list, indent=2)
    )


def format_gemma_manual(system_content: str, user_query: str, assistant_response: str) -> str:
    """
    Manually format using Gemma 4's <start_of_turn> template.
    Gemma has no separate system role — tool schemas are folded into the user turn.
    """
    return (
        f"<start_of_turn>user\n{system_content}\n\n{user_query}<end_of_turn>\n"
        f"<start_of_turn>model\n{assistant_response}<end_of_turn>\n"
    )


def try_load_tokenizer():
    """
    Attempt to load the Gemma 4 tokenizer from HF Hub.
    Returns the tokenizer on success, None on failure.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore

        hf_token = os.getenv("HF_TOKEN")
        print("Loading tokenizer google/gemma-4-e4b-it from HF Hub...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-4-e4b-it",
            token=hf_token,
        )
        print("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as exc:
        print(f"[WARN] Could not load tokenizer: {exc}")
        print("[WARN] Falling back to manual Gemma template formatting.")
        return None


def apply_template(
    tokenizer,
    system_content: str,
    user_query: str,
    assistant_response: str,
) -> str:
    """
    Format a single training example.

    Tries tokenizer.apply_chat_template first; falls back to manual formatting.
    """
    if tokenizer is not None:
        try:
            messages = [
                {"role": "user", "content": f"{system_content}\n\n{user_query}"},
                {"role": "assistant", "content": assistant_response},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as exc:
            print(f"[WARN] apply_chat_template failed ({exc}); using manual fallback.")

    return format_gemma_manual(system_content, user_query, assistant_response)


# ---------------------------------------------------------------------------
# JSON parsing / filtering
# ---------------------------------------------------------------------------

def parse_row(row: dict) -> dict | None:
    """
    Parse and validate a single dataset row.

    Returns a dict with parsed fields, or None if the row is invalid.
    """
    try:
        tools_list = json.loads(row["tools"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None

    try:
        answers_list = json.loads(row["answers"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None

    if not isinstance(tools_list, list):
        return None
    if not isinstance(answers_list, list) or len(answers_list) == 0:
        return None

    return {
        "query": row["query"],
        "tools_list": tools_list,
        "answers_list": answers_list,
        "tools_raw": row["tools"],
        "answers_raw": row["answers"],
    }


def format_answer(answers_list: list) -> str:
    """
    Format the answer field.
    - Single call: json.dumps of the first element
    - Multiple calls: json.dumps of the full list
    """
    if len(answers_list) == 1:
        return json.dumps(answers_list[0])
    return json.dumps(answers_list)


# ---------------------------------------------------------------------------
# Hash utilities
# ---------------------------------------------------------------------------

def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def save_jsonl(records: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Hash overlap check
# ---------------------------------------------------------------------------

def verify_no_overlap(train_path: Path, val_path: Path, test_path: Path) -> None:
    """
    Reload all three JSONL files, hash each text field, and assert no overlap.
    Raises AssertionError on failure; prints confirmation on success.
    """
    print("\nRunning hash overlap verification...")

    def get_hashes(path: Path) -> set:
        return {rec["_hash"] for rec in load_jsonl(path)}

    train_hashes = get_hashes(train_path)
    val_hashes = get_hashes(val_path)
    test_hashes = get_hashes(test_path)

    train_val = train_hashes & val_hashes
    train_test = train_hashes & test_hashes
    val_test = val_hashes & test_hashes

    if train_val or train_test or val_test:
        msg = (
            f"Hash overlap detected! "
            f"train∩val={len(train_val)}, "
            f"train∩test={len(train_test)}, "
            f"val∩test={len(val_test)}"
        )
        raise AssertionError(msg)

    print("Hash check PASSED — zero overlap between train, val, and test_held_out.")


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def stratified_split(
    single_call: list,
    multi_call: list,
    n_test: int,
    val_fraction: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Stratified split into (train, val, test_held_out).

    n_test examples are carved out first (proportional to class sizes),
    then the remainder is split val_fraction / (1 - val_fraction) into val/train.
    """
    rng = random.Random(seed)

    total = len(single_call) + len(multi_call)
    if total == 0:
        raise ValueError("No valid examples after filtering.")

    # Proportional test allocation
    n_test_single = round(n_test * len(single_call) / total)
    n_test_multi = n_test - n_test_single

    # Shuffle each stratum
    single_shuffled = single_call[:]
    multi_shuffled = multi_call[:]
    rng.shuffle(single_shuffled)
    rng.shuffle(multi_shuffled)

    # Carve out test
    test_single = single_shuffled[:n_test_single]
    rest_single = single_shuffled[n_test_single:]

    test_multi = multi_shuffled[:n_test_multi]
    rest_multi = multi_shuffled[n_test_multi:]

    test_held_out = test_single + test_multi

    # Split remainder into train / val (stratified)
    def split_remainder(items: list) -> tuple[list, list]:
        n_val = round(len(items) * val_fraction)
        return items[n_val:], items[:n_val]  # (train, val)

    train_single, val_single = split_remainder(rest_single)
    train_multi, val_multi = split_remainder(rest_multi)

    train = train_single + train_multi
    val = val_single + val_multi

    # Final shuffle of combined sets
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test_held_out)

    return train, val, test_held_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Salesforce/xlam-function-calling-60k for SFT on Gemma 4 E4B."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on valid (post-filter) examples (useful for quick tests).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory where train/val/test JSONL files will be written.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading Salesforce/xlam-function-calling-60k from HF Hub...")
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not found. Install it with: pip install datasets")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")
    ds = load_dataset(
        "Salesforce/xlam-function-calling-60k",
        split="train",
        token=hf_token,
    )
    print(f"Loaded {len(ds):,} rows.")

    # ------------------------------------------------------------------
    # 2. Load tokenizer (optional; fall back to manual template)
    # ------------------------------------------------------------------
    tokenizer = try_load_tokenizer()

    # ------------------------------------------------------------------
    # 3. Parse, filter, and convert rows
    # ------------------------------------------------------------------
    valid_records: list[dict] = []
    n_filtered = 0

    print("\nParsing and converting rows...")
    for row in tqdm(ds, desc="Converting", unit="rows"):
        parsed = parse_row(row)
        if parsed is None:
            n_filtered += 1
            continue

        system_content = build_system_content(parsed["tools_list"])
        assistant_response = format_answer(parsed["answers_list"])

        text = apply_template(
            tokenizer,
            system_content,
            parsed["query"],
            assistant_response,
        )

        is_multi = len(parsed["answers_list"]) > 1
        record = {
            "text": text,
            "query": parsed["query"],
            "tools_raw": parsed["tools_raw"],
            "answers_raw": parsed["answers_raw"],
            "is_multi_call": is_multi,
            "_hash": compute_text_hash(text),
        }
        valid_records.append(record)

    print(
        f"\nParsing complete: {len(valid_records):,} valid rows "
        f"({n_filtered:,} filtered out due to malformed JSON)."
    )

    # ------------------------------------------------------------------
    # 3a. Deduplicate by _hash (keep first occurrence)
    # ------------------------------------------------------------------
    seen_hashes: set[str] = set()
    deduped_records: list[dict] = []
    for rec in valid_records:
        if rec["_hash"] not in seen_hashes:
            seen_hashes.add(rec["_hash"])
            deduped_records.append(rec)
    n_duplicates = len(valid_records) - len(deduped_records)
    if n_duplicates:
        print(f"Deduplication removed {n_duplicates:,} duplicate rows.")
    valid_records = deduped_records

    # ------------------------------------------------------------------
    # 3b. Apply --max-samples cap on valid (post-filter) rows
    # ------------------------------------------------------------------
    if args.max_samples is not None and len(valid_records) > args.max_samples:
        print(f"Capping valid records to {args.max_samples:,} (--max-samples={args.max_samples}).")
        valid_records = valid_records[:args.max_samples]

    # Partition into stratification buckets
    single_call: list[dict] = [r for r in valid_records if not r["is_multi_call"]]
    multi_call: list[dict] = [r for r in valid_records if r["is_multi_call"]]

    total_valid = len(valid_records)
    print(f"  Single-call: {len(single_call):,}")
    print(f"  Multi-call:  {len(multi_call):,}")

    # ------------------------------------------------------------------
    # 4. Stratified split
    # ------------------------------------------------------------------
    n_test = 500
    val_fraction = 0.10  # 10% of remainder goes to val

    if total_valid <= n_test:
        n_test = max(0, total_valid // 5)  # fall back to 20% as held-out
        print(f"Warning: only {total_valid} valid examples; reducing test set to {n_test}")

    print(f"\nSplitting: {n_test} held-out test, 90/10 train/val of remainder...")
    train, val, test_held_out = stratified_split(
        single_call=single_call,
        multi_call=multi_call,
        n_test=n_test,
        val_fraction=val_fraction,
        seed=42,
    )

    # ------------------------------------------------------------------
    # 5. Save
    # ------------------------------------------------------------------
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test_held_out.jsonl"

    print(f"\nSaving files to {output_dir}/")
    save_jsonl(train, train_path)
    save_jsonl(val, val_path)
    save_jsonl(test_held_out, test_path)
    print(f"  train.jsonl         : {len(train):,} examples")
    print(f"  val.jsonl           : {len(val):,} examples")
    print(f"  test_held_out.jsonl : {len(test_held_out):,} examples")

    # ------------------------------------------------------------------
    # 6. Hash overlap verification
    # ------------------------------------------------------------------
    verify_no_overlap(train_path, val_path, test_path)

    # ------------------------------------------------------------------
    # 7. Summary stats
    # ------------------------------------------------------------------
    def count_multi(records: list) -> int:
        return sum(1 for r in records if r["is_multi_call"])

    print("\n=== Summary ===")
    print(f"Total rows loaded      : {len(ds):,}")
    print(f"Filtered (bad JSON)    : {n_filtered:,}")
    print(f"Valid rows             : {total_valid:,}")
    print(f"  Single-call          : {len(single_call):,}")
    print(f"  Multi-call           : {len(multi_call):,}")
    print(f"Train set              : {len(train):,} ({count_multi(train):,} multi-call)")
    print(f"Val set                : {len(val):,} ({count_multi(val):,} multi-call)")
    print(f"Test held-out          : {len(test_held_out):,} ({count_multi(test_held_out):,} multi-call)")
    print("\nDone.")


if __name__ == "__main__":
    main()
