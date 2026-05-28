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


def test_gemini_runner_cache_hit(tmp_path):
    """If output JSON exists, gemini_runner returns it without calling the API."""
    out = tmp_path / "results_gemini.json"
    out.write_text(json.dumps(_cached_result()))

    from eval import gemini_runner
    result = gemini_runner.run(output_path=str(out))
    assert result == _cached_result()


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
