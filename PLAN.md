# Project 3 — Fine-Tuned Tool-Use Specialist (Gemma 4 E4B)

## 0. The headline

You're an Agentic AI engineer. P1 showed you *orchestrate* agents. P2 showed you *build tools* for agents. P3 shows you *train a model* that's specifically good at using tools — by fine-tuning the latest open model (Gemma 4 E4B, April 2026) on the Salesforce xLAM-60k function-calling dataset.

**Total cost: $0.** Free Colab T4 for training. Free HF Hub for model hosting. Free Groq + Gemini AI Studio for eval baselines. Free everything.

**What you ship:**
- A fine-tuned `gemma-4-e4b-tool-use-lora` on Hugging Face Hub with a proper model card.
- An honest eval table comparing the fine-tune against base Gemma 4 E4B, Llama 3.3 70b (via Groq), and Gemini 2.0 Flash — all free.
- A GitHub repo with reproducible training and eval code.
- Optional Gradio demo on free HF Spaces.
- A LinkedIn post + blog post completing the P1 → P2 → P3 thesis.

## 2. Project overview

### What we're building
A fine-tuned 4B-parameter open model specialized in **structured tool/function calling** — emitting valid JSON function invocations given a user query and a list of available tool schemas.

### Success criteria
- [ ] Trained model published to `huggingface.co/sahilmdeshmukh/gemma-4-e4b-tool-use-lora`.
- [ ] Honest eval comparison on a 500-example held-out set across all four free systems.
- [ ] Fine-tune **meaningfully beats** Gemma 4 E4B base on AST-match.
- [ ] README has a Results table with real numbers + cost-per-call analysis.
- [ ] HF Hub model card is filled in properly.
- [ ] Repo reproducible: `make data && make train && make eval`.

## 3. Architecture / approach

### Output format
Each example provides a user query + available tool schemas (JSON Schema). The model outputs:
```json
{"name": "get_weather", "arguments": {"location": "Pune, India", "unit": "celsius"}}
```

### Dataset pipeline
xLAM-60k ships in Llama chat-template format. We convert to Gemma format:

**Gemma format:**
```
<start_of_turn>user
{system_message_with_tool_schemas}

{query}
<end_of_turn>
<start_of_turn>model
{tool_call_json}
<end_of_turn>
```

**Pipeline steps:**
1. `load_dataset("Salesforce/xlam-function-calling-60k")`
2. Filter malformed JSON answers
3. Convert: strip Llama markup, re-emit as message lists, apply Gemma 4 chat template
4. Split 90/5/5 + carve out 500 held-out examples (NEVER used in training)
5. Save as `data/train.jsonl`, `data/val.jsonl`, `data/test_held_out.jsonl`

### Training
- Base: `google/gemma-4-e4b-it` (4B params, Apache-2.0)
- Method: QLoRA — 4-bit base + LoRA adapters
- Hardware: Free Google Colab T4
- Hyperparams: rank=16, alpha=32, lr=2e-4, batch=2, grad_accum=8, epochs=1

## 4. Tech stack
- Base model: `google/gemma-4-e4b-it`
- Training: TRL SFTTrainer + PEFT (LoRA)
- Quantization: bitsandbytes 4-bit NF4
- Dataset: `Salesforce/xlam-function-calling-60k`
- Tracking: wandb
- Eval: Custom BFCL-style harness
- Baselines: Groq (Llama 3.3 70b) + Gemini AI Studio (Flash)
- Hub: Hugging Face Hub
- Env mgmt: uv

## 5. Day-by-day status
- [ ] Day 1: dataset prep (xLAM → Gemma chat-template format, splits)
- [ ] Day 2: QLoRA training + push to HF Hub
- [ ] Day 3: free-only eval harness
- [ ] Day 4: model card + README + (optional) Spaces demo
