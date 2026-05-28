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


def score_example(expected: dict | str, predicted: dict | list | None) -> dict:
    """Compute BFCL-style scores for one example."""
    if isinstance(expected, str):
        try:
            expected = json.loads(expected)
        except json.JSONDecodeError:
            pass
    if not isinstance(expected, dict):
        return {"name_match": 0, "args_key_match": 0.0, "args_value_match": 0.0}

    if predicted is None:
        return {"name_match": 0, "args_key_match": 0.0, "args_value_match": 0.0}

    if isinstance(predicted, list):
        predicted = predicted[0] if predicted else None
    if not isinstance(predicted, dict):
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
