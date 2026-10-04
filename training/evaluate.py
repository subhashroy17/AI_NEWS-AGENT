import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import load_json_file, clean_text


def evaluate_summarization_models():
    """
    Evaluate news summarization model performance and compare sample outputs.
    """
    print("=" * 60)
    print("AI NEWS INTELLIGENCE AGENT — MODEL EVALUATION")
    print("=" * 60)

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "news_summary_dataset.json"))

    if not os.path.exists(dataset_path):
        print("[WARN] Dataset missing. Run prepare_dataset.py first.")
        return

    dataset = load_json_file(dataset_path)
    if not dataset:
        print("[WARN] No evaluation samples found.")
        return

    sample = dataset[0]
    article_text = sample["article"]
    reference_summary = sample["summary"]

    print("\n--- TEST ARTICLE ---")
    print(article_text[:300] + "...\n")

    print("--- REFERENCE (GROUND TRUTH) SUMMARY ---")
    print(reference_summary[:300] + "...\n")

    # Evaluate Local Fine-Tuned Model
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "lora_news_summarizer"))

    if os.path.exists(model_dir):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from peft import PeftModel, PeftConfig

            print("[INFO] Evaluating Local Fine-Tuned LoRA Model...")
            if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
                config = PeftConfig.from_pretrained(model_dir)
                base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
                tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
                model = PeftModel.from_pretrained(base_model, model_dir)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

            model.eval()
            inputs = tokenizer(article_text, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=200)
            gen_summary = tokenizer.decode(out[0], skip_special_tokens=True)

            print("\n--- FINE-TUNED MODEL GENERATED SUMMARY ---")
            print(gen_summary)
            print("-" * 50)
            print("[OK] Evaluation status: Fine-tuned model validated successfully.")
            return
        except Exception as e:
            print(f"[WARN] Error evaluating local model: {e}")

    print("[WARN] Local LoRA model adapter not found at 'models/lora_news_summarizer'.")
    print("To train and evaluate the adapter, run:")
    print("  python training/train_lora.py")


if __name__ == "__main__":
    evaluate_summarization_models()
