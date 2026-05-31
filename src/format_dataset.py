"""
format_dataset.py
Converts raw NegTox DataFrame into HuggingFace instruction-tuning format.
Output format: {"prompt": ..., "completion": ...}
"""

import pandas as pd
from datasets import Dataset


SYSTEM_PROMPT = (
    "You are NegTox-LLM, a specialized biomedical AI for neglected tropical disease "
    "drug discovery. Given a molecule's descriptors and SMILES, predict its bioactivity "
    "against the specified disease and explain your reasoning."
)


def format_row(row: dict) -> dict:
    """
    Convert a single row into prompt/completion pair.
    """
    prompt = (
        f"### System:\n{SYSTEM_PROMPT}\n\n"
        f"### Input:\n{row['instruction']}\n\n"
        f"### Task:\nPredict bioactivity against {row['disease']}. "
        f"Answer Active or Inactive and explain why.\n\n"
        f"### Response:\n"
    )
    completion = (
        f"{row['output']}. This molecule has MW={row.get('MW', 'N/A')}, "
        f"LogP={row.get('LogP', 'N/A')}, TPSA={row.get('TPSA', 'N/A')}. "
        f"pChEMBL value: {row.get('pchembl_value', 'N/A')}."
    )
    return {"prompt": prompt, "completion": completion}


def dataframe_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Convert full DataFrame to HuggingFace Dataset object.
    """
    records = []
    for _, row in df.iterrows():
        formatted = format_row(row.to_dict())
        records.append(formatted)
    return Dataset.from_list(records)


def scaffold_split(df: pd.DataFrame,
                   train_ratio: float = 0.70,
                   val_ratio: float = 0.15) -> tuple:
    """
    Simple random split (scaffold split runs on Kaggle with deepchem).
    Returns train, val, test DataFrames.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df[:train_end], df[train_end:val_end], df[val_end:]


if __name__ == "__main__":
    # Sanity test
    sample = pd.DataFrame([{
        "instruction": "[Disease: Leishmaniasis][MW: 309.41][LogP: 2.76][TPSA: 35.58][HBD: 1][HBA: 3][RotBonds: 4][SMILES: CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)C3=CC=CC=C3]",
        "output": "Active",
        "disease": "Leishmaniasis",
        "pchembl_value": "6.5",
    }])
    dataset = dataframe_to_hf_dataset(sample)
    print(dataset[0]["prompt"])
    print("---")
    print(dataset[0]["completion"])