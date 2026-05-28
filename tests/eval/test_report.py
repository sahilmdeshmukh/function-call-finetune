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
