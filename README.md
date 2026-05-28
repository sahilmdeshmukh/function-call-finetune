# Tool-Use Specialist — Fine-tuned Gemma 4 E4B

> A 4B-parameter open model fine-tuned for structured tool/function calling. Built on Google's Gemma 4 (April 2026 release).


## Quickstart

```bash
git clone https://github.com/sahilmdeshmukh/function-call-finetune.git
cd function-call-finetune
uv sync
cp .env.example .env  # fill in GROQ_API_KEY + GEMINI_API_KEY
make data
make train
make eval
```

## Results

*(Will be filled in after Day 3 eval runs.)*

## License

MIT for code. Model weights inherit the Gemma 4 license (Apache-2.0).
