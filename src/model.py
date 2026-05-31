"""
model.py
Loads OLMo-7B with 4-bit QLoRA quantization for NegTox-LLM fine-tuning.
Runs on Kaggle T4 GPU (16GB VRAM).
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training


MODEL_NAME = "allenai/OLMo-7B-hf"


def get_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_lora_config() -> LoraConfig:
    """LoRA config — r=16, alpha=32, targeting attention projections."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def load_tokenizer(model_name: str = MODEL_NAME):
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name: str = MODEL_NAME, use_quantization: bool = True):
    """
    Load OLMo-7B with optional 4-bit quantization + LoRA.
    Set use_quantization=False for CPU testing.
    """
    bnb_config = get_bnb_config() if use_quantization else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if use_quantization:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, get_lora_config())
    model.print_trainable_parameters()
    return model


if __name__ == "__main__":
    print("Model config ready.")
    print(f"Target model: {MODEL_NAME}")
    print("Quantization: 4-bit NF4")
    print("LoRA: r=16, alpha=32")
    print("Trainable modules: q_proj, v_proj, k_proj, o_proj")