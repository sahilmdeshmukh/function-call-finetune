# Eval Harness Design — Day 3

## Goal

Compare 4 models on 200 held-out tool-calling examples to answer: did fine-tuning help?

Models:
1. Fine-tuned Gemma 2 2B (`Sahil3717/gemma-2-2b-tool-use-lora`) — GPU
2. Base Gemma 2 2B (`google/gemma-2-2b-it`) — GPU
3. Llama 3.3 70B via Groq free API
4. Gemini 2.0 Flash via Gemini AI Studio free API

## File Structure

```
eval/
  groq_runner.py       — 200 examples → Llama 3.3 70B → eval/results/results_groq.json
  gemini_runner.py     — 200 examples → Gemini 2.0 Flash → eval/results/results_gemini.json
  model_runner.py      — base + fine-tuned Gemma 2 2B → results_base.json / results_finetuned.json
  metrics.py           — BFCL scoring functions
  report.py            — merges 4 JSONs → results.md + printed summary

eval/results/          — all output files (gitignored except results.md)

notebooks/eval.ipynb   — orchestrator notebook, calls each runner then report.py
```

## Runner Contract

Every runner exposes:
```python
def run(test_path: str, n_samples: int = 200, output_path: str) -> list[dict]
```

Behaviour:
- If `output_path` already exists, load and return it (caching — safe on crash/resume)
- Sample `n_samples` examples from `test_path` using `random.seed(42)` (same 200 for all models)
- Run inference, parse output, score each example
- Save results array to `output_path` as JSON

Per-example result dict:
```python
{
  "query": str,
  "expected": dict,                  # parsed from answers_raw (first call only)
  "predicted_raw": str,              # raw model output text
  "predicted_parsed": dict | None,   # None if JSON parse fails
  "name_match": int,                 # 1 or 0
  "args_key_match": float,           # 0.0–1.0
  "args_value_match": float          # 0.0–1.0
}
```

## Metrics

Defined in `metrics.py`, applied per example:

- **name_match**: `int(predicted["name"] == expected["name"])` — exact string match
- **args_key_match**: fraction of expected argument keys present in prediction
- **args_value_match**: fraction of expected arg keys where value also matches exactly
- If JSON parse fails → all three scores = 0
- Multi-call examples (`is_multi_call=True`): compare first call only

## API Runners

Both `groq_runner.py` and `gemini_runner.py`:
- Use the same system prompt as training: tools JSON folded into the user turn
- Sleep 2 seconds between calls (safe for ~30 req/min free tier)
- Groq: `llama-3.3-70b-versatile` model
- Gemini: `gemini-2.0-flash` model

## Model Runner

`model_runner.py` runs both GPU models in one session to avoid loading the base model twice.
Caching is per output file — each is checked independently:

1. If both `results_base.json` and `results_finetuned.json` exist → skip entirely
2. Load base `google/gemma-2-2b-it` in 4-bit
3. If `results_base.json` missing → run 200 examples → save
4. Attach `Sahil3717/gemma-2-2b-tool-use-lora` adapter (PeftModel on top of loaded base)
5. If `results_finetuned.json` missing → run same 200 examples → save
6. Unload model

## Report

`report.py` reads all 4 result JSONs and produces:

1. **Printed summary table:**
```
Model                    | name_match | args_key | args_value
-------------------------|------------|----------|------------
Fine-tuned Gemma 2 2B    |    0.XX    |   0.XX   |    0.XX
Base Gemma 2 2B          |    0.XX    |   0.XX   |    0.XX
Llama 3.3 70B (Groq)     |    0.XX    |   0.XX   |    0.XX
Gemini 2.0 Flash         |    0.XX    |   0.XX   |    0.XX
```

2. **`eval/results/results.md`** — same table in markdown, ready to paste into README

## Notebook Flow

`notebooks/eval.ipynb` steps:
1. Install deps + load secrets (same as train.ipynb)
2. Clone/pull repo, cd into it
3. Run `groq_runner.run(...)` — skips if cached
4. Run `gemini_runner.run(...)` — skips if cached
5. Run `model_runner.run(...)` — loads GPU models, runs both
6. Run `report.generate(...)` — prints table, writes results.md
7. Commit results.md to git

## Constraints

- All free: Groq free tier, Gemini AI Studio free tier, Kaggle T4
- 200 examples: fits in rate limits, statistically meaningful
- Same 200 examples for all models (`random.seed(42)`)
- Results JSON committed to git; `results.md` included in README
