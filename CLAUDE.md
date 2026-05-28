# Claude Code Notes — Tool-Use Specialist SLM

## Read first
PLAN.md has the full project plan. Read it before any change.
Third project in a three-part agentic-AI thesis:
- P1 (multi-agent-research-analyst): I orchestrate agents
- P2 (india-pharma-mcp): I build tools agents use
- P3 (this): I trained a model to use tools better

## Stack
Python 3.12 · uv · Hugging Face Transformers · PEFT (LoRA/QLoRA) ·
TRL (SFTTrainer) · bitsandbytes (4-bit) · datasets · wandb · pydantic
Eval baselines via Groq (free) and Gemini AI Studio (free).

## Base model
google/gemma-2-2b-it (2B params, text-only, Apache-2.0)
Switched from Gemma 4 E4B — the multimodal variant was ~8B effective params and too slow on T4.
Gemma 2 2B is text-only, trains in ~1-2 hrs on free T4. Chat template: <start_of_turn>user / <start_of_turn>model.

## Dataset
Salesforce/xlam-function-calling-60k (Hugging Face).
Note: xLAM is formatted for Llama-style chat templates. We MUST convert
it to Gemma's <start_of_turn> format before SFT. See PLAN.md §3.

## Cost discipline
Total project cost target: $0. No paid APIs.
- Training: free Colab T4 / Kaggle P100
- Eval baselines: Groq (free) + Gemini AI Studio (free)
- Model hosting: HF Hub (free)
- Demo: HF Spaces (free tier)

## Conventions
- Conventional commits: feat:, fix:, refactor:, docs:, test:, chore:
- Atomic commits — one logical change per commit
- All training configs in YAML, not hard-coded
- Eval results committed as JSON; results.md generated from them
- BFCL-style metrics: function-name match, args-key match, args-value match

## Status
- [x] Day 1: dataset prep (xLAM → Gemma chat-template format, splits)
- [x] Day 2: QLoRA training + push to HF Hub (Sahil3717/gemma-2-2b-tool-use-lora)
- [ ] Day 3: free-only eval harness
- [ ] Day 4: model card + README + (optional) Spaces demo

## Open questions
(empty — add as they come up)
