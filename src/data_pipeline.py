"""
data_pipeline.py
Downloads NTD compounds from ChEMBL, cleans SMILES,
computes descriptors, builds instruction-tuning dataset.
"""

import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from src.descriptors import process_molecule
from typing import Optional


DISEASE_TARGETS = {
    "Leishmaniasis": "Leishmania",
    "Chagas":        "Trypanosoma cruzi",
    "Malaria":       "Plasmodium falciparum",
    "Tuberculosis":  "Mycobacterium tuberculosis",
}


def fetch_chembl_compounds(organism: str, max_records: int = 5000) -> pd.DataFrame:
    """
    Fetch bioactive compounds for a given organism from ChEMBL.
    Returns DataFrame with SMILES, activity, and metadata.
    """
    activity = new_client.activity
    results = activity.filter(
        target_organism__icontains=organism,
        standard_type__in=["IC50", "Ki", "EC50"],
        relation="=",
    ).only([
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_value",
        "standard_units",
        "standard_type",
        "pchembl_value",
    ])[:max_records]

    df = pd.DataFrame(list(results))
    return df


def clean_smiles(smiles: str) -> Optional[str]:
    """Validate and canonicalize SMILES. Returns None if invalid."""
    if not isinstance(smiles, str) or smiles.strip() == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def assign_label(pchembl_value) -> Optional[int]:
    """
    Binary activity label based on pChEMBL value.
    pChEMBL >= 6.0 (IC50 <= 1uM) → active (1), else inactive (0).
    """
    try:
        val = float(pchembl_value)
        return 1 if val >= 6.0 else 0
    except (TypeError, ValueError):
        return None


def build_dataset(disease: str, max_records: int = 5000) -> pd.DataFrame:
    """
    Full pipeline: fetch → clean → label → augment with descriptors.
    Returns instruction-tuning ready DataFrame.
    """
    organism = DISEASE_TARGETS[disease]
    print(f"[{disease}] Fetching from ChEMBL: {organism}...")
    df = fetch_chembl_compounds(organism, max_records)
    print(f"[{disease}] Raw records: {len(df)}")

    df["clean_smiles"] = df["canonical_smiles"].apply(clean_smiles)
    df["label"] = df["pchembl_value"].apply(assign_label)
    df = df.dropna(subset=["clean_smiles", "label"])
    df["label"] = df["label"].astype(int)
    print(f"[{disease}] After cleaning: {len(df)} valid molecules")

    df["instruction"] = df.apply(
        lambda row: process_molecule(row["clean_smiles"], disease), axis=1
    )
    df = df.dropna(subset=["instruction"])

    df["disease"] = disease
    df["output"] = df["label"].map({1: "Active", 0: "Inactive"})

    return df[[
        "molecule_chembl_id", "disease", "clean_smiles",
        "instruction", "label", "output", "pchembl_value"
    ]]


if __name__ == "__main__":
    # Quick test — 100 molecules only
    df = build_dataset("Leishmaniasis", max_records=100)
    print(df[["instruction", "output"]].head(3))
    print(f"\nShape: {df.shape}")