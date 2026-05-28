# Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a BFCL-style eval harness that compares 4 models on 200 tool-calling examples and produces a results table in JSON + Markdown.

**Architecture:** Modular runners (`eval/groq_runner.py`, `eval/gemini_runner.py`, `eval/model_runner.py`) each cache their own results JSON. A shared `eval/metrics.py` scores every example. `eval/report.py` merges all 4 JSONs and prints + writes `eval/results/results.md`. `notebooks/eval.ipynb` orchestrates everything.

**Tech Stack:** Python 3.12, groq, google-generativeai, transformers, peft, bitsandbytes, pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `eval/__init__.py` | Package marker (empty) |
| `eval/prompt.py` | Shared `SYSTEM_PROMPT_TEMPLATE` |
| `eval/metrics.py` | `parse_prediction()`, `score_example()` |
| `eval/groq_runner.py` | Llama 3.3 70B via Groq API → `results_groq.json` |
| `eval/gemini_runner.py` | Gemini 2.0 Flash → `results_gemini.json` |
| `eval/model_runner.py` | Base + fine-tuned Gemma 2 2B → `results_base.json` / `results_finetuned.json` |
| `eval/report.py` | Merge 4 JSONs → printed table + `results.md` |
| `eval/results/` | Output directory (JSON files gitignored, results.md committed) |
| `tests/eval/test_metrics.py` | Unit tests for metrics |
| `tests/eval/test_runners_cache.py` | Cache-hit tests for all runners |
| `tests/eval/test_report.py` | Tests for report generation |
| `notebooks/eval.ipynb` | Orchestrator notebook for Kaggle T4 |

---

## Task 1: Scaffold

**Files:**
- Create: `eval/__init__.py`
- Create: `eval/prompt.py`
- Create: `eval/results/.gitkeep`
- Create: `tests/__init__.py`
- Create: `tests/eval/__init__.py`
- Modify: `.gitignore`

- [ ] **Step 1: Create eval package and shared prompt**

`eval/__init__.py` — empty file.

`eval/prompt.py`:
```python
SYSTEM_PROMPT_TEMPLATE = (
    "You have access to the following functions. Use them if required:\n\n"
    "{tools_json}\n\n"
    'To call a function, respond with a JSON object of the format:\n'
    '{{"name": "function_name", "arguments": {{"key": "value"}}}}\n'
    "For multiple calls, respond with a JSON array of such objects."
)
```

- [ ] **Step 2: Create test package and results directory**

`tests/__init__.py` — empty file.
`tests/eval/__init__.py` — empty file.
`eval/results/.gitkeep` — empty file (ensures directory is tracked by git).

- [ ] **Step 3: Update .gitignore**

Add to the end of `.gitignore`:
```
# Eval results JSON (large — commit results.md only)
eval/results/*.json
```

- [ ] **Step 4: Commit**

```bash
git add eval/ tests/ .gitignore
git commit -m "chore: scaffold eval package, test dirs, results gitignore"
```

---

## Task 2: metrics.py (TDD)

**Files:**
- Create: `eval/metrics.py`
- Create: `tests/eval/test_metrics.py`

- [ ] **Step 1: Write failing tests**

`tests/eval/test_metrics.py`:
```python
import pytest
from eval.metrics import parse_prediction, score_example


def test_parse_valid_json():
    raw = '{"name": "get_weather", "arguments": {"location": "Mumbai"}}'
    result = parse_prediction(raw)
    assert result == {"name": "get_weather", "arguments": {"location": "Mumbai"}}


def test_parse_json_in_code_block():
    raw = '```json\n{"name": "get_weather", "arguments": {"location": "Mumbai"}}\n```'
    result = parse_prediction(raw)
    assert result is not None
    assert result["name"] == "get_weather"


def test_parse_json_with_surrounding_text():
    raw = 'Sure! {"name": "get_weather", "arguments": {"city": "Delhi"}} done.'
    result = parse_prediction(raw)
    assert result is not None
    assert result["name"] == "get_weather"


def test_parse_invalid_returns_none():
    assert parse_prediction("I cannot help.") is None
    assert parse_prediction("") is None
    assert parse_prediction("{invalid}") is None


def test_score_perfect_match():
    expected  = {"name": "get_weather", "arguments": {"location": "Mumbai", "unit": "celsius"}}
    predicted = {"name": "get_weather", "arguments": {"location": "Mumbai", "unit": "celsius"}}
    s = score_example(expected, predicted)
    assert s["name_match"] == 1
    assert s["args_key_match"] == 1.0
    assert s["args_value_match"] == 1.0


def test_score_name_mismatch():
    expected  = {"name": "get_weather", "arguments": {"location": "Mumbai"}}
    predicted = {"name": "get_temperature", "arguments": {"location": "Mumbai"}}
    s = score_example(expected, predicted)
    assert s["name_match"] == 0
    assert s["args_key_match"] == 1.0


def test_score_partial_keys():
    expected  = {"name": "fn", "arguments": {"a": "1", "b": "2", "c": "3"}}
    predicted = {"name": "fn", "arguments": {"a": "1", "b": "2"}}
    s = score_example(expected, predicted)
    assert pytest.approx(s["args_key_match"])   == 2 / 3
    assert pytest.approx(s["args_value_match"]) == 2 / 3


def test_score_key_present_value_wrong():
    expected  = {"name": "fn", "arguments": {"location": "Mumbai"}}
    predicted = {"name": "fn", "arguments": {"location": "Delhi"}}
    s = score_example(expected, predicted)
    assert s["args_key_match"]   == 1.0
    assert s["args_value_match"] == 0.0


def test_score_none_prediction():
    s = score_example({"name": "fn", "arguments": {"x": "1"}}, None)
    assert s == {"name_match": 0, "args_key_match": 0.0, "args_value_match": 0.0}


def test_score_no_args():
    expected  = {"name": "fn", "arguments": {}}
    predicted = {"name": "fn", "arguments": {}}
    s = score_example(expected, predicted)
    assert s["args_key_match"] == 1.0
    assert s["args_value_match"] == 1.0
```

- [ ] **Step 2: Run tests — expect FAIL (ImportError)**

```bash
pytest tests/eval/test_metrics.py -v
```
Expected: `ImportError: cannot import name 'parse_prediction' from 'eval.metrics'`

- [ ] **Step 3: Implement eval/metrics.py**

```python
import json
import re


def parse_prediction(raw: str) -> dict | None:
    """Parse model output as JSON. Returns None on failure."""
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def score_example(expected: dict, predicted: dict | None) -> dict:
    """Compute BFCL-style scores for one example."""
    if predicted is None:
        return {"name_match": 0, "args_key_match": 0.0, "args_value_match": 0.0}

    name_match = int(predicted.get("name") == expected.get("name"))
    exp_args   = expected.get("arguments", {})
    pred_args  = predicted.get("arguments", {})

    if not exp_args:
        return {"name_match": name_match, "args_key_match": 1.0, "args_value_match": 1.0}

    present = [k for k in exp_args if k in pred_args]
    matching = [k for k in present if str(pred_args[k]) == str(exp_args[k])]

    return {
        "name_match":       name_match,
        "args_key_match":   len(present)  / len(exp_args),
        "args_value_match": len(matching) / len(exp_args),
    }
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
pytest tests/eval/test_metrics.py -v
```
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/metrics.py tests/eval/test_metrics.py
git commit -m "feat: add metrics.py — parse_prediction + score_example (TDD)"
```

---

## Task 3: groq_runner.py

**Files:**
- Create: `eval/groq_runner.py`
- Create: `tests/eval/test_runners_cache.py` (first test)

- [ ] **Step 1: Write the cache-hit test**

`tests/eval/test_runners_cache.py`:
```python
import json
import pytest
from pathlib import Path


def _cached_result():
    return [{"query": "q", "expected": {}, "predicted_raw": "",
             "predicted_parsed": None, "name_match": 0,
             "args_key_match": 0.0, "args_value_match": 0.0}]


def test_groq_runner_cache_hit(tmp_path):
    """If output JSON exists, groq_runner returns it without calling the API."""
    out = tmp_path / "results_groq.json"
    out.write_text(json.dumps(_cached_result()))

    from eval import groq_runner
    result = groq_runner.run(output_path=str(out))
    assert result == _cached_result()
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/eval/test_runners_cache.py::test_groq_runner_cache_hit -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement eval/groq_runner.py**

```python
import json
import os
import random
import time
from pathlib import Path

from eval.metrics import parse_prediction, score_example
from eval.prompt import SYSTEM_PROMPT_TEMPLATE


def run(
    test_path: str = "data/test_held_out.jsonl",
    n_samples: int = 200,
    output_path: str = "eval/results/results_groq.json",
    seed: int = 42,
) -> list[dict]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"[groq] Cache hit — loading {out}")
        with open(out) as f:
            return json.load(f)

    records = _load_jsonl(test_path)
    samples = random.Random(seed).sample(records, min(n_samples, len(records)))

    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    results = []
    for i, rec in enumerate(samples):
        tools    = json.loads(rec["tools_raw"])
        expected = json.loads(rec["answers_raw"])[0]
        prompt   = SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))

        try:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"{prompt}\n\n{rec['query']}"}],
                temperature=0.0,
                max_tokens=256,
            )
            predicted_raw = resp.choices[0].message.content
        except Exception as e:
            print(f"[groq] Error on {i}: {e}")
            predicted_raw = ""

        predicted_parsed = parse_prediction(predicted_raw)
        results.append({
            "query":            rec["query"],
            "expected":         expected,
            "predicted_raw":    predicted_raw,
            "predicted_parsed": predicted_parsed,
            **score_example(expected, predicted_parsed),
        })

        if (i + 1) % 25 == 0:
            print(f"[groq] {i + 1}/{len(samples)}")
        time.sleep(2)

    _save(results, out)
    return results


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[groq] Saved {len(results)} results to {path}")
```

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/eval/test_runners_cache.py::test_groq_runner_cache_hit -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/groq_runner.py tests/eval/test_runners_cache.py
git commit -m "feat: add groq_runner — Llama 3.3 70B eval with caching"
```

---

## Task 4: gemini_runner.py

**Files:**
- Create: `eval/gemini_runner.py`
- Modify: `tests/eval/test_runners_cache.py` (add test)

- [ ] **Step 1: Add cache-hit test for Gemini**

Append to `tests/eval/test_runners_cache.py`:
```python
def test_gemini_runner_cache_hit(tmp_path):
    """If output JSON exists, gemini_runner returns it without calling the API."""
    out = tmp_path / "results_gemini.json"
    out.write_text(json.dumps(_cached_result()))

    from eval import gemini_runner
    result = gemini_runner.run(output_path=str(out))
    assert result == _cached_result()
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/eval/test_runners_cache.py::test_gemini_runner_cache_hit -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement eval/gemini_runner.py**

```python
import json
import os
import random
import time
from pathlib import Path

import google.generativeai as genai

from eval.metrics import parse_prediction, score_example
from eval.prompt import SYSTEM_PROMPT_TEMPLATE


def run(
    test_path: str = "data/test_held_out.jsonl",
    n_samples: int = 200,
    output_path: str = "eval/results/results_gemini.json",
    seed: int = 42,
) -> list[dict]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():
        print(f"[gemini] Cache hit — loading {out}")
        with open(out) as f:
            return json.load(f)

    records = _load_jsonl(test_path)
    samples = random.Random(seed).sample(records, min(n_samples, len(records)))

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-2.0-flash")

    results = []
    for i, rec in enumerate(samples):
        tools    = json.loads(rec["tools_raw"])
        expected = json.loads(rec["answers_raw"])[0]
        prompt   = SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))

        try:
            resp = model.generate_content(f"{prompt}\n\n{rec['query']}")
            predicted_raw = resp.text
        except Exception as e:
            print(f"[gemini] Error on {i}: {e}")
            predicted_raw = ""

        predicted_parsed = parse_prediction(predicted_raw)
        results.append({
            "query":            rec["query"],
            "expected":         expected,
            "predicted_raw":    predicted_raw,
            "predicted_parsed": predicted_parsed,
            **score_example(expected, predicted_parsed),
        })

        if (i + 1) % 25 == 0:
            print(f"[gemini] {i + 1}/{len(samples)}")
        time.sleep(2)

    _save(results, out)
    return results


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _save(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gemini] Saved {len(results)} results to {path}")
```

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/eval/test_runners_cache.py::test_gemini_runner_cache_hit -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/gemini_runner.py tests/eval/test_runners_cache.py
git commit -m "feat: add gemini_runner — Gemini 2.0 Flash eval with caching"
```

---

## Task 5: model_runner.py

**Files:**
- Create: `eval/model_runner.py`
- Modify: `tests/eval/test_runners_cache.py` (add test)

- [ ] **Step 1: Add cache-hit test for model runner**

Append to `tests/eval/test_runners_cache.py`:
```python
def test_model_runner_cache_hit_both(tmp_path):
    """If both output files exist, model_runner loads without touching the GPU."""
    base_data = [{"name_match": 1, "args_key_match": 1.0, "args_value_match": 1.0}]
    ft_data   = [{"name_match": 1, "args_key_match": 0.5, "args_value_match": 0.5}]

    base_out = tmp_path / "results_base.json"
    ft_out   = tmp_path / "results_finetuned.json"
    base_out.write_text(json.dumps(base_data))
    ft_out.write_text(json.dumps(ft_data))

    from eval import model_runner
    base_r, ft_r = model_runner.run(
        base_output_path=str(base_out),
        finetuned_output_path=str(ft_out),
    )
    assert base_r == base_data
    assert ft_r   == ft_data
```

- [ ] **Step 2: Run test — expect FAIL**

```bash
pytest tests/eval/test_runners_cache.py::test_model_runner_cache_hit_both -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement eval/model_runner.py**

```python
import gc
import json
import os
import random
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from eval.metrics import parse_prediction, score_example
from eval.prompt import SYSTEM_PROMPT_TEMPLATE

BASE_MODEL_ID = "google/gemma-2-2b-it"
ADAPTER_ID    = "Sahil3717/gemma-2-2b-tool-use-lora"


def run(
    test_path: str = "data/test_held_out.jsonl",
    n_samples: int = 200,
    base_output_path: str = "eval/results/results_base.json",
    finetuned_output_path: str = "eval/results/results_finetuned.json",
    base_model_id: str = BASE_MODEL_ID,
    adapter_id: str = ADAPTER_ID,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    base_path = Path(base_output_path)
    ft_path   = Path(finetuned_output_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)

    base_done = base_path.exists()
    ft_done   = ft_path.exists()

    if base_done and ft_done:
        print("[model] Both cached — loading from disk")
        return _load(base_path), _load(ft_path)

    samples  = _load_samples(test_path, n_samples, seed)
    hf_token = os.getenv("HF_TOKEN")

    gc.collect()
    torch.cuda.empty_cache()

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb, device_map="auto", token=hf_token
    )

    if not base_done:
        print("[model] Running base model inference...")
        base_results = _infer(model, tokenizer, samples)
        _save(base_results, base_path)
    else:
        print("[model] Base cached — skipping")
        base_results = _load(base_path)

    if not ft_done:
        print("[model] Attaching fine-tuned adapter...")
        ft_model = PeftModel.from_pretrained(model, adapter_id, token=hf_token)
        ft_results = _infer(ft_model, tokenizer, samples)
        _save(ft_results, ft_path)
        del ft_model
    else:
        print("[model] Fine-tuned cached — skipping")
        ft_results = _load(ft_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return base_results, ft_results


def _load_samples(test_path: str, n_samples: int, seed: int) -> list[dict]:
    with open(test_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return random.Random(seed).sample(records, min(n_samples, len(records)))


def _infer(model, tokenizer, samples: list[dict]) -> list[dict]:
    model.eval()
    results = []
    for i, rec in enumerate(samples):
        tools    = json.loads(rec["tools_raw"])
        expected = json.loads(rec["answers_raw"])[0]
        prompt   = SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
        messages = [{"role": "user", "content": f"{prompt}\n\n{rec['query']}"}]

        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(inputs, max_new_tokens=128, temperature=0.1, do_sample=True)

        predicted_raw    = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        predicted_parsed = parse_prediction(predicted_raw)

        results.append({
            "query":            rec["query"],
            "expected":         expected,
            "predicted_raw":    predicted_raw,
            "predicted_parsed": predicted_parsed,
            **score_example(expected, predicted_parsed),
        })

        if (i + 1) % 20 == 0:
            print(f"[model] {i + 1}/{len(samples)}")
    return results


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def _save(results: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[model] Saved {len(results)} results to {path}")
```

- [ ] **Step 4: Run test — expect PASS**

```bash
pytest tests/eval/test_runners_cache.py::test_model_runner_cache_hit_both -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/model_runner.py tests/eval/test_runners_cache.py
git commit -m "feat: add model_runner — base + fine-tuned Gemma 2 2B with per-file caching"
```

---

## Task 6: report.py

**Files:**
- Create: `eval/report.py`
- Create: `tests/eval/test_report.py`

- [ ] **Step 1: Write failing tests**

`tests/eval/test_report.py`:
```python
import json
from pathlib import Path


def _write_results(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _make_result(name_match=1, args_key=0.8, args_value=0.6):
    return {
        "query": "q", "expected": {}, "predicted_raw": "", "predicted_parsed": None,
        "name_match": name_match, "args_key_match": args_key, "args_value_match": args_value,
    }


def test_report_generates_markdown(tmp_path):
    for fname in ["results_finetuned.json", "results_base.json",
                  "results_groq.json", "results_gemini.json"]:
        _write_results(tmp_path / fname, [_make_result()])

    out_md = tmp_path / "results.md"
    from eval.report import generate
    generate(results_dir=str(tmp_path), output_md=str(out_md))

    assert out_md.exists()
    content = out_md.read_text()
    assert "Fine-tuned Gemma 2 2B" in content
    assert "1.000" in content   # name_match = 1.0
    assert "0.800" in content   # args_key = 0.8


def test_report_skips_missing_files(tmp_path, capsys):
    _write_results(tmp_path / "results_groq.json", [_make_result()])
    out_md = tmp_path / "out.md"

    from eval.report import generate
    generate(results_dir=str(tmp_path), output_md=str(out_md))

    captured = capsys.readouterr()
    assert "Missing" in captured.out
    assert out_md.exists()


def test_report_averages_correctly(tmp_path):
    data = [_make_result(name_match=1), _make_result(name_match=0)]
    _write_results(tmp_path / "results_finetuned.json", data)
    out_md = tmp_path / "out.md"

    from eval.report import generate
    generate(results_dir=str(tmp_path), output_md=str(out_md))

    content = out_md.read_text()
    assert "0.500" in content   # avg name_match = 0.5
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
pytest tests/eval/test_report.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement eval/report.py**

```python
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
```

- [ ] **Step 4: Run all tests — expect PASS**

```bash
pytest tests/eval/ -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add eval/report.py tests/eval/test_report.py
git commit -m "feat: add report.py — merge results, print table, write results.md"
```

---

## Task 7: eval.ipynb (Kaggle orchestrator notebook)

**Files:**
- Create: `notebooks/eval.ipynb`

- [ ] **Step 1: Create the notebook**

Create `notebooks/eval.ipynb` with these cells (use `NotebookEdit` tool):

**Cell: intro (markdown)**
```
# Eval Harness — Tool-Use Specialist

Compares 4 models on 200 held-out tool-calling examples:
- Fine-tuned Gemma 2 2B (Sahil3717/gemma-2-2b-tool-use-lora)
- Base Gemma 2 2B (google/gemma-2-2b-it)
- Llama 3.3 70B via Groq (free)
- Gemini 2.0 Flash via Gemini AI Studio (free)

**Before running:** Set Accelerator → GPU T4 x1. Add secrets: HF_TOKEN, GROQ_API_KEY, GEMINI_API_KEY.

Run cells top to bottom. Each runner caches its results — safe to re-run after a crash.
```

**Cell: check-gpu (code)**
```python
import subprocess
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print(f'GPU: {result.stdout.strip()}')
else:
    print('No GPU — go to Settings → Accelerator → GPU T4 x1')
```

**Cell: install (code)**
```python
%%capture
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q \
    peft>=0.12.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.31.0 \
    datasets>=2.19.0 \
    huggingface-hub>=0.23.0 \
    groq>=0.9.0 \
    google-generativeai>=0.7.0 \
    sentencepiece>=0.2.0 \
    protobuf>=5.27.0 \
    python-dotenv>=1.0.0

import transformers, groq, google.generativeai as genai
print(f'transformers {transformers.__version__}')
print('All dependencies installed')
```

**Cell: secrets (code)**
```python
import os

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
    from kaggle_secrets import UserSecretsClient
    s = UserSecretsClient()
    os.environ['HF_TOKEN']       = s.get_secret('HF_TOKEN')
    os.environ['GROQ_API_KEY']   = s.get_secret('GROQ_API_KEY')
    os.environ['GEMINI_API_KEY'] = s.get_secret('GEMINI_API_KEY')
    print('Running on Kaggle')
else:
    from google.colab import userdata
    os.environ['HF_TOKEN']       = userdata.get('HF_TOKEN')
    os.environ['GROQ_API_KEY']   = userdata.get('GROQ_API_KEY')
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')
    print('Running on Colab')

for key in ['HF_TOKEN', 'GROQ_API_KEY', 'GEMINI_API_KEY']:
    print(f'{"OK" if os.environ.get(key) else "MISSING"} {key}')
```

**Cell: clone (code)**
```python
import os
from pathlib import Path

REPO     = 'function-call-finetune'
REPO_URL = 'https://github.com/sahilmdeshmukh/function-call-finetune.git'

if Path(REPO).exists():
    print('Repo exists — pulling...')
    !git -C {REPO} pull
else:
    !git clone {REPO_URL}

os.chdir(REPO)
print(f'Working dir: {os.getcwd()}')
```

**Cell: run-groq (code)**
```python
from eval.groq_runner import run as groq_run

groq_results = groq_run(
    test_path="data/test_held_out.jsonl",
    n_samples=200,
    output_path="eval/results/results_groq.json",
)
print(f"Groq: {len(groq_results)} examples done")
```

**Cell: run-gemini (code)**
```python
from eval.gemini_runner import run as gemini_run

gemini_results = gemini_run(
    test_path="data/test_held_out.jsonl",
    n_samples=200,
    output_path="eval/results/results_gemini.json",
)
print(f"Gemini: {len(gemini_results)} examples done")
```

**Cell: run-models (code)**
```python
from eval.model_runner import run as model_run

base_results, ft_results = model_run(
    test_path="data/test_held_out.jsonl",
    n_samples=200,
    base_output_path="eval/results/results_base.json",
    finetuned_output_path="eval/results/results_finetuned.json",
)
print(f"Base: {len(base_results)} | Fine-tuned: {len(ft_results)} examples done")
```

**Cell: report (code)**
```python
from eval.report import generate

generate(
    results_dir="eval/results",
    output_md="eval/results/results.md",
)
```

**Cell: commit (code)**
```python
!git add eval/results/results.md
!git commit -m "feat: add eval results — Day 3 complete"
!git push
print("Results committed and pushed!")
```

- [ ] **Step 2: Commit notebook**

```bash
git add notebooks/eval.ipynb
git commit -m "feat: add eval.ipynb — orchestrator notebook for Kaggle T4"
```

---

## Task 8: Final push and verify

- [ ] **Step 1: Run the full test suite**

```bash
pytest tests/eval/ -v
```
Expected: all tests PASS. If any fail, fix before proceeding.

- [ ] **Step 2: Push to GitHub**

```bash
git push
```

- [ ] **Step 3: Verify on GitHub**

Check that `github.com/sahilmdeshmukh/function-call-finetune` shows:
- `eval/` directory with 6 Python files
- `tests/eval/` with 3 test files
- `notebooks/eval.ipynb`
- `eval/results/.gitkeep` (but no `.json` files — they're gitignored)
