"""
evaluate.py
Evaluation pipeline for NegTox-LLM.
Metrics: ROC-AUC per disease, ablation table, known drug sanity checks.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.descriptors import process_molecule


KNOWN_DRUGS = {
    "Leishmaniasis": [
        ("CC(=O)OCC(=O)[C@@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@@]12C", "Active"),
    ],
    "Chagas": [
        ("C1CN(CCN1CC2=CC=CC=N2)C(=O)C3=CC=C(C=C3)[N+](=O)[O-]", "Active"),
    ],
    "Malaria": [
        ("CCN1CCN(CC1)CC2=CC=CC=C2Cl", "Active"),
    ],
}


def predict_single(
    smiles: str,
    disease: str,
    model,
    tokenizer,
    max_new_tokens: int = 20,
) -> str:
    """
    Run inference on a single molecule.
    Returns raw model output string.
    """
    instruction = process_molecule(smiles, disease)
    if instruction is None:
        return "Invalid SMILES"

    prompt = (
        f"### Input:\n{instruction}\n\n"
        f"### Task:\nPredict bioactivity against {disease}. Answer Active or Inactive.\n\n"
        f"### Response:\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("### Response:\n")[-1].strip()
    return response


def parse_label(response: str) -> int:
    """Extract binary label from model response."""
    response_lower = response.lower()
    if "active" in response_lower and "inactive" not in response_lower:
        return 1
    return 0


def evaluate_dataset(
    df: pd.DataFrame,
    model,
    tokenizer,
    disease_col: str = "disease",
    smiles_col: str = "clean_smiles",
    label_col: str = "label",
) -> dict:
    """
    Run evaluation on full test set.
    Returns ROC-AUC per disease + overall.
    """
    results = {}
    df = df.copy()
    df["predicted"] = df.apply(
        lambda row: parse_label(
            predict_single(row[smiles_col], row[disease_col], model, tokenizer)
        ),
        axis=1,
    )

    for disease in df[disease_col].unique():
        subset = df[df[disease_col] == disease]
        if len(subset["label"].unique()) < 2:
            continue
        auc = roc_auc_score(subset[label_col], subset["predicted"])
        results[disease] = round(auc, 4)
        print(f"[{disease}] ROC-AUC: {auc:.4f}")

    overall_auc = roc_auc_score(df[label_col], df["predicted"])
    results["overall"] = round(overall_auc, 4)
    print(f"\nOverall ROC-AUC: {overall_auc:.4f}")
    return results


def sanity_check(model, tokenizer):
    """Test on known NTD drugs — must predict Active."""
    print("\n--- Sanity Check: Known Drugs ---")
    for disease, molecules in KNOWN_DRUGS.items():
        for smiles, expected in molecules:
            response = predict_single(smiles, disease, model, tokenizer)
            predicted = "Active" if parse_label(response) == 1 else "Inactive"
            status = "✓" if predicted == expected else "✗"
            print(f"{status} [{disease}] Expected: {expected} | Got: {predicted}")


if __name__ == "__main__":
    print("evaluate.py ready — run from Kaggle after training.")