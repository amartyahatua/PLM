from transformers import pipeline
import numpy as np
import torch
import pandas as pd
import math
from tqdm import tqdm

#better for small dataset or single sequence
def evaluate(test_df, model, tokenizer):
    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    correct = 0
    total = 0

    top1_correct = 0
    top5_correct = 0
    total = 0
    log_probs = []

    for seq in test_df['Sequence']:
        if len(seq) < 5:
            continue
        pos = torch.randint(1, len(seq)-1, (1,)).item()
        true_token = seq[pos]
        seq_masked = list(seq)
        seq_masked[pos] = tokenizer.mask_token
        masked_input = " ".join(seq_masked)
        masked_input = f"{tokenizer.cls_token} {masked_input} {tokenizer.sep_token}"

        try:
            preds = fill_mask(masked_input)
            top_preds = [p['token_str'].strip() for p in preds]
            if true_token == top_preds[0]:
                top1_correct += 1
            if true_token in top_preds:
                top5_correct += 1
            total += 1

            # Get the predicted log-prob of the original token at position i
            prob = next((p["score"] for p in preds if p["token_str"] == true_token), 1e-9)
            log_probs.append(np.log(prob))
        except:
            continue

    top_1_acc = top1_correct / total
    top_5_acc = top5_correct / total
    perplexity = np.exp(-np.mean(log_probs))

    print(f"Top-1 Accuracy: {top_1_acc:.4f}")
    print(f"Top-5 Accuracy: {top_5_acc:.4f}")
    print(f"Pseudo-Perplexity: {perplexity:.4f}")

    return top_1_acc, top_5_acc, perplexity

def evaluate_from_full_sequence_batched(df, model, tokenizer, batch_size=32):
    """
    Evaluates a masked language model using batched inputs for efficiency on GPU.

    Assumes each row in the DataFrame contains a full sentence, and the target word is the last one.
    """
    # Load model and tokenizer
    nlp = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    mask_token = tokenizer.mask_token

    # Prepare masked inputs and labels
    masked_texts = []
    true_labels = []

    for _, row in df.iterrows():
        full_sequence = row['Sequence'].strip()
        tokens = full_sequence.split()

        if len(tokens) < 1:
            continue

        true_label = tokens[-1].strip(".,!?;:")
        masked_tokens = tokens[:-1] + [mask_token + '.']  # Add back period for structure
        masked_text = " ".join(masked_tokens)

        masked_texts.append(masked_text)
        true_labels.append(true_label)

    # Initialize metrics
    top1_correct = 0
    top5_correct = 0
    total_log_probs = []

    # Process in batches
    for i in tqdm(range(0, len(masked_texts), batch_size), desc="Evaluating"):
        batch_texts = masked_texts[i:i+batch_size]
        batch_labels = true_labels[i:i+batch_size]

        try:
            batch_predictions = nlp(batch_texts)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            continue

        for preds, label in zip(batch_predictions, batch_labels):
            predicted_tokens = [pred['token_str'].strip() for pred in preds[:5]]

            if label == predicted_tokens[0]:
                top1_correct += 1
            if label in predicted_tokens:
                top5_correct += 1

            matched = next((pred for pred in preds if pred['token_str'].strip() == label), None)
            prob = matched['score'] if matched else 1e-10
            total_log_probs.append(-np.log(prob))

    n = len(true_labels)
    top1_acc = top1_correct / n
    top5_acc = top5_correct / n
    perplexity = math.exp(np.mean(total_log_probs))

    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'perplexity': perplexity
    }

