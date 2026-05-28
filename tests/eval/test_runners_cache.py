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
