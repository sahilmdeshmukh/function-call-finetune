import json
import os
import random
import time
from pathlib import Path

from google import genai

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

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    results = []
    for i, rec in enumerate(samples):
        tools    = json.loads(rec["tools_raw"])
        _ans     = json.loads(rec["answers_raw"])[0]
        expected = json.loads(_ans) if isinstance(_ans, str) else _ans
        prompt   = SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))

        try:
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"{prompt}\n\n{rec['query']}",
            )
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
