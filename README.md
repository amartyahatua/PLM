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
