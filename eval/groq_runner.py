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
