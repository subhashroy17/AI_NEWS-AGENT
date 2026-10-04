# Models Directory

This directory stores trained LoRA/QLoRA adapter weights for domain-specific news summarization.

## Expected Directory Structure

```
models/
└── lora_news_summarizer/
    ├── adapter_config.json
    ├── adapter_model.safetensors (or adapter_model.bin)
    └── README.md
```

## Loading Logic

When the AI News Intelligence Agent starts:
1. It checks if `models/lora_news_summarizer` contains a valid trained adapter model.
2. If found, the application loads the fine-tuned model as the **PRIMARY** summarizer.
3. If not found or if system hardware is insufficient, the system gracefully falls back to the **Groq API**.

## How to Train

To train your own LoRA adapter on custom news data:

```bash
python training/prepare_dataset.py
python training/train_lora.py
python training/evaluate.py
```
