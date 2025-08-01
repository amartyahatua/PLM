# 🧬 Protein Language Modeling (PLM)

This repository enables fine-tuning of Transformer-based models like **RoBERTa**, **ESM**, and **ProtBERT** on protein sequence data from Swiss‑Prot. It supports both **Masked Language Modeling (MLM)** and **Enzyme Commission (EC) number classification** tasks.

---

## 🚀 Table of Contents

- [Features](#-features)  
- [Installation](#-installation)  
- [File-structure](#-File-structure)  
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

## File-structure
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

## EC Number Classification

python driver.py \
  --train_csv_path ./SwissprotDatasets/BalancedSwissprot/train.csv \
  --test_csv_path ./SwissprotDatasets/BalancedSwissprot/valid.csv \
  --model roberta-base \
  --tokenizer roberta-base \
  --output_dir ./results \
  --mlm_probability 0.15 \
  --epochs 3 \
  --per_device_train_batch_size 16 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 100
