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
