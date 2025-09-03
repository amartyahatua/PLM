#!/bin/bash

#SBATCH --job-name=lookingglassv3_test
#SBATCH --partition=hoarfrost_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=4-00:00:00
#SBATCH --output=jobs/%x_%j.out
#SBATCH --error=jobs/%x_%j.err
#SBATCH --mail-user=Ashley.Babjac@uga.edu
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

ml Python/3.11.3-GCCcore-12.3.0
ml CUDA/12.1.1

source /scratch/ab18558/PLM/env/bin/activate 

python mask_prediction/driver.py \
  --train_csv_path ./data/data/Swissprot/BalancedSwissprot/train.csv \
  --test_csv_path ./data/data/Swissprot/BalancedSwissprot/valid.csv \
  --model roberta-base \
  --tokenizer roberta-base \
  --output_dir ./test/ \
  --mlm_probability 0.15 \
  --epochs 1 \
  --per_device_train_batch_size 16 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --logging_steps 100
