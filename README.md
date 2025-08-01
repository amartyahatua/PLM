# 🧬 Protein Language Modeling (PLM)

This repository enables fine-tuning of Transformer-based models like **RoBERTa**, **ESM**, and **ProtBERT** on protein sequence data from Swiss‑Prot. It supports both **Masked Language Modeling (MLM)** and **Enzyme Commission (EC) number classification** tasks.

---

## 🚀 Table of Contents

- [Features](#-features)  
- [Installation](#-installation)  
- [Data Structure](#-data-structure)  
- [Available Scripts](#-available-scripts)  
  - EC Classification  
  - Masked LM Finetuning  
- [Usage Examples](#-usage-examples)  
- [Troubleshooting & Tips](#-troubleshooting--tips)  
- [Acknowledgments](#-acknowledgments)  
- [License](#-license)

---

## ✨ Features

- **Preprocess protein sequences** into space-separated format for language models.
- **MLM fine-tuning** with Optuna hyperparameter search support.
- **EC number classification**, with model training and evaluation pipelines.
- Hyperparameter tuning with **custom search spaces** using Optuna.
- Configurable via `argparse` for flexible experimentation.

---

## 🧰 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/amartyahatua/PLM.git
cd PLM
pip install -r requirements.txt

PLM/
│
├── dataset.py               # Data-loading utilities
├── data_prepcess.py         # Sequence formatting and spacing
├── hp_finetune.py           # Hyperparameter search space definition
├── driver.py                # Entry point for MLM training + HP tuning
├── ec_classification.py     # Script for EC classification training & evaluation
├── mask_model.py            # Encapsulates the MaskModel class
├── requirements.txt         # Project dependencies
│
└── SwissprotDatasets/
    └── BalancedSwissprot/
        ├── train.csv        # Balanced Swiss‑Prot sequences + EC labels
        └── valid.csv        # Validation set with similar structure

