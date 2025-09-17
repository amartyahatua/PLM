#!/bin/bash

#SBATCH --job-name=evaluation_ModernBERT
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128gb
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/%x_%j.out
#SBATCH --error=jobs/%x_%j.err
#SBATCH --mail-user=Ashley.Babjac@uga.edu
#SBATCH --mail-type=ALL

cd $SLURM_SUBMIT_DIR

ml Python/3.11.3-GCCcore-12.3.0
ml CUDA/12.1.1

#default is answerdotai/ModernBERT-base
#--model facebook/esm2_t30_150M_UR50D \
#--model roberta-base

source /scratch/ab18558/PLM/env/bin/activate 

for n in {1..5};
do
	python mask_prediction/test.py \
  	--test_csv_path ./data/data/GTDB/old/GTDB_dataset/Amartya_small/split_$n/test.pkl \
#	--model roberta-base \
#	--tokenizer roberta-base \
  	--output_dir ./hyperparams/ModernBERT/split_$n/ 
done
