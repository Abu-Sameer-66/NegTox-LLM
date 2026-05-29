<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:000000,30:001a0e,60:002d1a,100:003d24&height=320&section=header&text=🧬%20NegTox-LLM&fontSize=80&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=World's%20First%20Descriptor-Augmented%20LLM%20for%20NTD%20Drug%20Discovery&descSize=18&descAlignY=60&descAlign=50&descColor=00FFB3" width="100%"/>

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&weight=700&size=24&pause=1000&color=00FFB3&center=true&vCenter=true&width=900&lines=World's+First+Open-Source+LLM+for+NTD+Drug+Discovery;OLMo-7B+%2B+QLoRA+%2B+Descriptor+Augmentation;4+Diseases+%7C+1.7B+People+%7C+5+Novel+Contributions;Leishmaniasis+%C2%B7+Chagas+%C2%B7+Malaria+%C2%B7+Tuberculosis" />

<br/><br/>

<a href="https://github.com/Abu-Sameer-66/NegTox-LLM">
<img src="https://img.shields.io/badge/%F0%9F%9F%A2%20REPO-GitHub-00FFB3?style=for-the-badge&labelColor=001a0e"/>
</a>
&nbsp;
<a href="https://huggingface.co/Abu-Sameer-66">
<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-HuggingFace-00D4FF?style=for-the-badge&labelColor=001a0e"/>
</a>
&nbsp;
<a href="https://www.kaggle.com/sameernadeem66">
<img src="https://img.shields.io/badge/%F0%9F%93%8A%20Notebooks-Kaggle-00FFB3?style=for-the-badge&logo=kaggle&labelColor=002d1a"/>
</a>
&nbsp;
<a href="https://www.linkedin.com/in/sameer-nadeem-66339a357/">
<img src="https://img.shields.io/badge/LinkedIn-Sameer%20Nadeem-00D4FF?style=for-the-badge&logo=linkedin&labelColor=001a0e"/>
</a>

<br/><br/>

<img src="https://img.shields.io/badge/Python-3.10-00FFB3?style=flat-square&logo=python&logoColor=black&labelColor=001a0e"/>
<img src="https://img.shields.io/badge/OLMo--7B-QLoRA-00D4FF?style=flat-square&labelColor=002d1a"/>
<img src="https://img.shields.io/badge/RDKit-2026.3-00FFB3?style=flat-square&labelColor=001a0e"/>
<img src="https://img.shields.io/badge/DeepChem-2.7-00D4FF?style=flat-square&labelColor=002d1a"/>
<img src="https://img.shields.io/badge/ChEMBL-API%20Live-00FFB3?style=flat-square&labelColor=001a0e"/>
<img src="https://img.shields.io/badge/Kaggle-T4%20GPU-00D4FF?style=flat-square&logo=kaggle&labelColor=002d1a"/>
<img src="https://img.shields.io/badge/WandB-Tracked-00FFB3?style=flat-square&labelColor=001a0e"/>

</div>

---

## The Disease Nobody Is Solving

**1.7 billion people suffer from Neglected Tropical Diseases. The global AI revolution has not reached them.**

That number is not a rounding error. Leishmaniasis, Chagas disease, Malaria, and Tuberculosis collectively devastate the world's poorest populations — and the drug discovery pipeline for these diseases is decades behind.

The problem is not a lack of computing power. The problem is that **AI drug discovery tools are built for profit, not people.**

Big pharma has no financial incentive. Academic ML benchmarks focus on cancer and Alzheimer's. ChEMBL contains millions of NTD bioactivity records that have never been fed into a large language model. KDD 2025 confirmed that descriptor-augmented LLMs remain entirely unexplored. Frontiers Chemistry 2021 — still open in 2026 — documented that AI advances have not reached NTD drug discovery pipelines.

**NegTox-LLM is the first system that closes this gap.**

---

## What Nobody Combined Before — 5 Novel Contributions

| # | Contribution | Why It Matters |
|:---|:---|:---|
| 1 | Instruction-tuned 7B LLM for NTDs | First ever — no GNN, no RF, a reasoning model |
| 2 | Descriptor-augmented input format | `[Disease][MW][LogP][TPSA][HBD][HBA][SMILES]` — KDD 2025 confirmed gap |
| 3 | Dual-head output | Bioactivity + Tox21 12-endpoint toxicity simultaneously |
| 4 | NTD-specific training corpus | ChEMBL data filtered to 4 diseases — not general chemistry |
| 5 | Free + open on Kaggle | Accessible to Global South researchers, zero cost |

---

## NegTox-LLM vs Everything Before It

| Approach | Method | Input | Coverage | Explainable | Free |
|:---|:---:|:---:|:---:|:---:|:---:|
| Legacy ML (2000s) | Random Forest / SVM | Raw SMILES | One disease | ❌ | ✅ |
| GNN Models | Graph Neural Networks | Molecular graph | General chemistry | ❌ | ✅ |
| Raw SMILES LLM | Base LLM | SMILES only | General | Partial | ✅ |
| Commercial tools | Proprietary | Varies | Varies | ❌ | ❌ |
| **NegTox-LLM** | **OLMo-7B + QLoRA** | **SMILES + 6 descriptors** | **4 NTDs simultaneously** | **✅** | **✅** |

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                     INPUT LAYER                      │
│    SMILES string  +  Target Disease  (user query)    │
└────────────────────────┬─────────────────────────────┘
                         │
          descriptor augmentation
          MW · LogP · TPSA · HBD · HBA · RotBonds
                         │
┌────────────────────────▼─────────────────────────────┐
│           INSTRUCTION FORMAT BUILDER                 │
│  [Disease: X][MW: X][LogP: X][TPSA: X][SMILES: X]   │
└────────────────────────┬─────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────┐
│              OLMo-7B + QLoRA (4-bit NF4)             │
│                                                      │
│  Training  →  ChEMBL NTD compounds (50k molecules)   │
│  Hardware  →  Kaggle T4 GPU (16GB VRAM, free)        │
│  Tracking  →  WandB experiment logs                  │
│  LoRA      →  r=16, alpha=32, q/v/k/o projections    │
└────────────────────────┬─────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
┌─────────────────┐           ┌─────────────────────┐
│  BIOACTIVITY    │           │  TOXICITY HEAD      │
│  Active /       │           │  Tox21 12-endpoint  │
│  Inactive       │           │  simultaneous       │
│  per disease    │           │  prediction         │
└─────────────────┘           └─────────────────────┘
```

---

## Quick Start

```bash
git clone https://github.com/Abu-Sameer-66/NegTox-LLM.git
cd NegTox-LLM
conda create -n negtox python=3.10 -y
conda activate negtox
pip install -r requirements.txt
```

### Run descriptor pipeline locally
```python
from src.descriptors import process_molecule

result = process_molecule("CCO", "Leishmaniasis")
print(result)
# [Disease: Leishmaniasis][MW: 46.07][LogP: -0.14][TPSA: 20.23][HBD: 1][HBA: 1][RotBonds: 0][SMILES: CCO]
```

### Run data pipeline (fetches from ChEMBL)
```python
from src.data_pipeline import build_dataset

df = build_dataset("Leishmaniasis", max_records=500)
print(df[["instruction", "output"]].head())
```

> Full training runs on Kaggle T4 GPU. See `notebooks/04_training.ipynb`.

---

## Diseases Covered

| Disease | Pathogen | People Affected | ChEMBL Target |
|:---|:---|:---:|:---|
| Leishmaniasis | *Leishmania* spp. | 350M+ at risk | Leishmania |
| Chagas Disease | *Trypanosoma cruzi* | 6–7M infected | Trypanosoma cruzi |
| Malaria | *Plasmodium falciparum* | 600M+ cases/yr | Plasmodium falciparum |
| Tuberculosis | *Mycobacterium tuberculosis* | 10M new cases/yr | Mycobacterium tuberculosis |

---

## Training Pipeline

```
Stage 1 — Sanity Check      →  500 molecules,  1 epoch   →  verify no NaN loss
Stage 2 — Validation Run    →  5,000 molecules, 3 epochs →  ROC-AUC > 0.65 target
Stage 3 — Full Training     →  50,000 molecules, 5 epochs →  production model
Stage 4 — Ablation Study    →  no descriptors baseline   →  prove augmentation helps
```

### Ablation Table (target results)

| Model Variant | Leishmaniasis | Chagas | Malaria | TB | Avg ROC-AUC |
|:---|:---:|:---:|:---:|:---:|:---:|
| DeepChem baseline | — | — | — | — | ~0.72 |
| OLMo-7B raw SMILES | — | — | — | — | TBD |
| **NegTox-LLM (ours)** | — | — | — | — | **TBD** |

*Results will be updated after Kaggle training runs.*

---

## Research Gap Evidence

| Source | Year | Gap Identified |
|:---|:---:|:---|
| KDD 2025 | 2025 | "Handcrafted molecular descriptors are underexplored in LLMs" |
| Frontiers Chemistry | 2021 | AI advances have not reached NTD drug discovery |
| Nature Communications | May 2026 | NTDs still lack automated AI pipelines |
| WHO NTD Report | 2024 | Zero open-source LLM exists for NTD drug discovery |

---

## Project Structure

```
NegTox-LLM/
├── data/
│   ├── raw/               ← ChEMBL downloaded records
│   ├── processed/         ← cleaned + descriptor-augmented
│   └── splits/            ← train / val / test scaffold splits
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_training.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_demo.ipynb
├── src/
│   ├── descriptors.py     ← MW, LogP, TPSA, HBD, HBA, RotBonds
│   ├── data_pipeline.py   ← ChEMBL → clean → label → augment
│   ├── format_dataset.py  ← instruction-tuning format builder
│   ├── model.py           ← OLMo-7B + QLoRA loader
│   ├── train.py           ← training loop + WandB logging
│   └── evaluate.py        ← ROC-AUC, ablation, sanity checks
├── app/
│   └── app.py             ← Streamlit demo (HuggingFace Spaces)
├── configs/
│   └── training_config.yaml
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Roadmap

- [x] Project architecture designed
- [x] Conda environment configured (`negtox`, Python 3.10)
- [x] Core source files: `descriptors.py`, `data_pipeline.py`, `format_dataset.py`, `model.py`, `train.py`, `evaluate.py`
- [x] Descriptor pipeline verified: RDKit + ChEMBL operational
- [ ] ChEMBL data collection — 4 diseases, 50k molecules
- [ ] Preprocessing + scaffold splits uploaded to HuggingFace Hub
- [ ] Sanity training run — 500 molecules, 1 epoch, Kaggle T4
- [ ] Validation run — 5k molecules, ROC-AUC > 0.65
- [ ] Full training — 50k molecules, 5 epochs
- [ ] Ablation study — descriptor augmentation vs raw SMILES
- [ ] Streamlit app deployed on HuggingFace Spaces
- [ ] arXiv preprint submitted

---

## Author

<div align="center">

**Sameer Nadeem** — AI/ML Engineer · Data Scientist · Open Source Contributor · GSoC 2026

<br/>

<a href="https://sameer-nadeem-portfolio.vercel.app"><img src="https://img.shields.io/badge/Portfolio-Live-00FFB3?style=for-the-badge&labelColor=001a0e"/></a>
&nbsp;
<a href="https://github.com/Abu-Sameer-66"><img src="https://img.shields.io/badge/GitHub-Abu--Sameer--66-00D4FF?style=for-the-badge&logo=github&labelColor=002d1a"/></a>
&nbsp;
<a href="https://www.linkedin.com/in/sameer-nadeem-66339a357/"><img src="https://img.shields.io/badge/LinkedIn-Sameer%20Nadeem-00FFB3?style=for-the-badge&logo=linkedin&labelColor=001a0e"/></a>
&nbsp;
<a href="https://www.kaggle.com/sameernadeem66"><img src="https://img.shields.io/badge/Kaggle-sameernadeem66-00D4FF?style=for-the-badge&logo=kaggle&labelColor=002d1a"/></a>
&nbsp;
<a href="https://huggingface.co/Abu-Sameer-66"><img src="https://img.shields.io/badge/HuggingFace-Abu--Sameer--66-00FFB3?style=for-the-badge&labelColor=001a0e"/></a>

</div>

---

<div align="center">

*Built for the 1.7 billion people big pharma forgot.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:003d24,50:002d1a,100:000000&height=140&section=footer" width="100%"/>

</div>
