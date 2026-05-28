import json
from pathlib import Path

MODELS = [
    ("results_finetuned.json", "Fine-tuned Gemma 2 2B"),
    ("results_base.json",      "Base Gemma 2 2B"),
    ("results_groq.json",      "Llama 3.3 70B (Groq)"),
    ("results_gemini.json",    "Gemini 2.0 Flash"),
]


def generate(
    results_dir: str = "eval/results",
    output_md: str = "eval/results/results.md",
) -> None:
    results_dir = Path(results_dir)
    rows = []

    for filename, label in MODELS:
        path = results_dir / filename
        if not path.exists():
            print(f"[report] Missing {path} — skipping")
            continue
        with open(path) as f:
            data = json.load(f)
        rows.append({
            "label":            label,
            "name_match":       _avg(data, "name_match"),
            "args_key_match":   _avg(data, "args_key_match"),
            "args_value_match": _avg(data, "args_value_match"),
            "n":                len(data),
        })

    _print_table(rows)
    _write_markdown(rows, output_md)


def _avg(data: list[dict], key: str) -> float:
    return sum(r[key] for r in data) / len(data) if data else 0.0


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'Model':<30} | {'name_match':>10} | "
        f"{'args_key':>10} | {'args_value':>10} | {'n':>5}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for r in rows:
        print(
            f"{r['label']:<30} | {r['name_match']:>10.3f} | "
            f"{r['args_key_match']:>10.3f} | {r['args_value_match']:>10.3f} | {r['n']:>5}"
        )
    print(f"{sep}\n")


def _write_markdown(rows: list[dict], output_md: str) -> None:
    n = rows[0]["n"] if rows else 0
    lines = [
        "# Eval Results\n",
        f"n={n} held-out examples, same random seed (42) across all models.\n",
        "| Model | name_match | args_key_match | args_value_match |",
        "|-------|-----------|---------------|-----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['name_match']:.3f} | "
            f"{r['args_key_match']:.3f} | {r['args_value_match']:.3f} |"
        )
    Path(output_md).parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] Written to {output_md}")
