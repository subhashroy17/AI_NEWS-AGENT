import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from typing import Optional
from utils.helpers import load_json_file


def train_lora_news_summarizer(
    base_model_name: str = "google/flan-t5-small",
    output_dir: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 2,
    lr: float = 5e-4,
):
    """
    Train a parameter-efficient LoRA adapter on news summarization data using HuggingFace PEFT.
    """
    print("=" * 60)
    print("AI NEWS INTELLIGENCE AGENT — LoRA FINE-TUNING PIPELINE")
    print("=" * 60)

    # 1. Path Setup
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "lora_news_summarizer"))
    output_dir = os.path.abspath(output_dir)

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "news_summary_dataset.json"))

    if not os.path.exists(dataset_path):
        print("[INFO] Training dataset missing. Running prepare_dataset.py first...")
        from training.prepare_dataset import create_sample_news_training_dataset
        create_sample_news_training_dataset()

    raw_data = load_json_file(dataset_path)
    print(f"[OK] Loaded {len(raw_data)} training samples from: {dataset_path}")

    # 2. Imports for PyTorch & Transformers / PEFT
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        print(f"[ERROR] Missing required ML dependencies: {e}")
        print("Please install requirements: pip install torch transformers peft accelerate datasets")
        return

    # 3. Load Pretrained Tokenizer & Model
    print(f"[INFO] Loading base pretrained model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

    # 4. Configure LoRA (Parameter-Efficient Fine-Tuning)
    print("[INFO] Applying LoRA adapter config...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. Tokenize Dataset
    def preprocess_function(examples):
        inputs = [item["article"] for item in examples]
        targets = [item["summary"] for item in examples]

        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=256, truncation=True, padding="max_length")

        labels_ids = labels["input_ids"]
        # Replace padding token id's with -100 so they are ignored by loss
        labels_ids = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels_ids]
        model_inputs["labels"] = labels_ids
        return model_inputs

    processed_data = preprocess_function(raw_data)

    # Convert to PyTorch Dataset structure
    class CustomNewsDataset(torch.utils.data.Dataset):
        def __init__(self, data_dict):
            self.input_ids = torch.tensor(data_dict["input_ids"])
            self.attention_mask = torch.tensor(data_dict["attention_mask"])
            self.labels = torch.tensor(data_dict["labels"])

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

    train_dataset = CustomNewsDataset(processed_data)

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=1,
        save_strategy="epoch",
        use_cpu=not torch.cuda.is_available(),
        report_to="none",
    )

    # 7. Initialize Trainer & Execute Fine-Tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("[INFO] Starting LoRA adapter training...")
    trainer.train()

    # 8. Save Trained LoRA Adapter & Tokenizer
    print(f"[INFO] Saving LoRA adapter weights to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("=" * 60)
    print("[OK] LoRA FINE-TUNING COMPLETED SUCCESSFULLY!")
    print(f"Model adapter saved at: {output_dir}")
    print("The Streamlit app will now detect and use this model as PRIMARY summarizer.")
    print("=" * 60)


if __name__ == "__main__":
    train_lora_news_summarizer()
