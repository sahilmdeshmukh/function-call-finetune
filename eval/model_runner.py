import gc
import json
import os
import random
from pathlib import Path

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

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

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
    import traceback
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    results = []
    for i, rec in enumerate(samples):
        tools    = json.loads(rec["tools_raw"])
        _ans     = json.loads(rec["answers_raw"])[0]
        expected = json.loads(_ans) if isinstance(_ans, str) else _ans
        prompt   = SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
        messages = [{"role": "user", "content": f"{prompt}\n\n{rec['query']}"}]

        try:
            inputs = tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(device)

            with torch.no_grad():
                out = model.generate(
                    inputs,
                    max_new_tokens=128,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            predicted_raw = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        except Exception as e:
            if i == 0:
                traceback.print_exc()
            else:
                print(f"[model] Error on {i}: {type(e).__name__}: {e}")
            predicted_raw = ""

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
