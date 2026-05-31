"""
train.py
QLoRA fine-tuning loop for NegTox-LLM on Kaggle T4 GPU.
Stages: sanity (500) → validation (5k) → full (50k) → ablation
"""

import os
import torch
import wandb
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from src.model import load_tokenizer, load_model


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int = 512) -> Dataset:
    """Tokenize prompt+completion pairs."""
    def tokenize(example):
        full_text = example["prompt"] + example["completion"]
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return dataset.map(tokenize, batched=False, remove_columns=dataset.column_names)


def get_training_args(
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 2e-4,
    run_name: str = "negtox-llm",
) -> TrainingArguments:
    """Build HuggingFace TrainingArguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=run_name,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
    )


def train(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str = "./outputs/negtox-model",
    epochs: int = 5,
    run_name: str = "negtox-llm-full",
    use_quantization: bool = True,
):
    """
    Main training function.
    Call from Kaggle notebook after loading datasets.
    """
    wandb.init(project="NegTox-LLM", name=run_name)

    tokenizer = load_tokenizer()
    model = load_model(use_quantization=use_quantization)

    train_tok = tokenize_dataset(train_dataset, tokenizer)
    eval_tok  = tokenize_dataset(eval_dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = get_training_args(
        output_dir=output_dir,
        epochs=epochs,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    wandb.finish()


if __name__ == "__main__":
    print("train.py ready — run from Kaggle notebook with GPU.")