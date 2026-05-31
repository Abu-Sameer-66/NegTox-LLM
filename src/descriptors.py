"""
descriptors.py
Computes molecular descriptors for NegTox-LLM input augmentation.
Descriptors: MW, LogP, TPSA, HBD, HBA, RotBonds
"""

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from typing import Optional


def compute_descriptors(smiles: str) -> Optional[dict]:
    """
    Compute 6 molecular descriptors from a SMILES string.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "MW":       round(Descriptors.MolWt(mol), 2),
        "LogP":     round(Descriptors.MolLogP(mol), 2),
        "TPSA":     round(rdMolDescriptors.CalcTPSA(mol), 2),
        "HBD":      rdMolDescriptors.CalcNumHBD(mol),
        "HBA":      rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }


def build_instruction(smiles: str, disease: str, descriptors: dict) -> str:
    """
    Build descriptor-augmented instruction string for LLM input.
    Format: [Disease: X][MW: X][LogP: X][TPSA: X][HBD: X][HBA: X][RotBonds: X][SMILES: X]
    """
    return (
        f"[Disease: {disease}]"
        f"[MW: {descriptors['MW']}]"
        f"[LogP: {descriptors['LogP']}]"
        f"[TPSA: {descriptors['TPSA']}]"
        f"[HBD: {descriptors['HBD']}]"
        f"[HBA: {descriptors['HBA']}]"
        f"[RotBonds: {descriptors['RotBonds']}]"
        f"[SMILES: {smiles}]"
    )


def process_molecule(smiles: str, disease: str) -> Optional[str]:
    """
    Full pipeline: SMILES → descriptors → instruction string.
    Returns None if SMILES is invalid.
    """
    descriptors = compute_descriptors(smiles)
    if descriptors is None:
        return None
    return build_instruction(smiles, disease, descriptors)


if __name__ == "__main__":
    # Quick sanity check
    test_smiles = "CC1=CC=C(C=C1)NC(=O)CN2CCN(CC2)C3=CC=CC=C3"  # Miltefosine-like
    result = process_molecule(test_smiles, "Leishmaniasis")
    print("Sample instruction:")
    print(result)