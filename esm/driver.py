import torch
import argparse
import numpy as np
from mask_model import MaskModel

RANDOM_STATE_SEED = 1829873
np.random.seed(RANDOM_STATE_SEED)
torch.manual_seed(RANDOM_STATE_SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EC Prediction Training Script")
    parser.add_argument("--train_csv_path", type=str, default="../data/train.csv", help="Path to the training CSV file")
    parser.add_argument("--test_csv_path", type=str, default="../data/valid.csv", help="Path to the test CSV file" )
    parser.add_argument("--tokenizer", type=str, default="facebook/esm1b_t33_650M_UR50S", help="Path to the test CSV file")
    parser.add_argument("--model", type=str, default="facebook/esm1b_t33_650M_UR50S", help="Path to the test CSV file")
    parser.add_argument("--output_dir", type=str, default="./facebook/esm1b_t33_650M_UR50S", help="output directory" )
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs" )
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--save_strategy", type=str, default="epoch", help="save strategy")
    parser.add_argument("--eval_strategy", type=str, default="no", help="eval strategy")
    parser.add_argument("--logging_steps", type=int, default=20, help="logging steps")
    parser.add_argument("--mlm_probability", type=int, default=0.15, help="mlm probability")


    # setup stuff
    # ----------------------------------------------------------------------------------#
    args = parser.parse_args()
    msk_model = MaskModel(args)
    msk_model.get_dataset()
    msk_model.call_trainer()
