#!/bin/bash

# Exit if any command fails
set -e

# Training configuration
TRAIN_CSV_PATH="../data/train.csv"
TEST_CSV_PATH="../data/valid.csv"
TOKENIZER="FacebookAI/roberta-base"
MODEL="FacebookAI/roberta-base"
OUTPUT_DIR="../output_roberta"
EPOCHS=3
BATCH_SIZE=4
SAVE_STRATEGY="epoch"
EVAL_STRATEGY="no"
LOGGING_STEPS=20
MLM_PROB=0.15

# Run the training script
python3 driver.py \
  --train_csv_path "$TRAIN_CSV_PATH" \
  --test_csv_path "$TEST_CSV_PATH" \
  --tokenizer "$TOKENIZER" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --epochs $EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --save_strategy "$SAVE_STRATEGY" \
  --eval_strategy "$EVAL_STRATEGY" \
  --logging_steps $LOGGING_STEPS \
  --mlm_probability $MLM_PROB
