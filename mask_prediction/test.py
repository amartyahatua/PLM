from evaluation import evaluate_from_full_sequence_batched
import pandas as pd
from transformers import AutoTokenizer, AutoModelForMaskedLM, EsmTokenizer, EsmForMaskedLM
import argparse
import os
import json
import numpy as np
import torch

os.environ["WANDB_DISABLED"] = "true"
RANDOM_STATE_SEED = 1829873
np.random.seed(RANDOM_STATE_SEED)
torch.manual_seed(RANDOM_STATE_SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')

def load_dataset(test_data_path):
    if test_data_path.__contains__('.csv'):
        test_df = pd.read_csv(test_data_path)
    elif test_data_path.__contains__('.tsv'):
        test_df = pd.read_csv(test_data_path, sep='\t')
    else:
        test_df = pd.read_pickle(test_data_path).rename(columns={'sequence' : 'Sequence'})

    return test_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EC Prediction Evaluation Script")
    parser.add_argument("--test_csv_path", type=str, default="../data/test.csv", help="Path to the test CSV file" )
    parser.add_argument("--tokenizer", type=str, default="answerdotai/ModernBERT-base", help="Path to the test CSV file")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base", help="Path to the test CSV file")
    parser.add_argument("--output_dir", type=str, default="../answerdotai/ModernBERT-base", help="output directory" )

    args = parser.parse_args()

    if 'esm' in args.model:
        tokenizer = EsmTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
        model = EsmForMaskedLM.from_pretrained(args.model, device_map='auto')
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
        model = AutoModelForMaskedLM.from_pretrained(args.model, device_map='auto')

    test_df = load_dataset(args.test_csv_path)

    metrics = evaluate_from_full_sequence_batched(test_df, model, tokenizer)

    with open(args.output_dir+'/test_metrics.json', 'w') as f:
        json.dump(metrics, f)
