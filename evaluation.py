from transformers import pipeline

# Step 14: Evaluation

fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

correct = 0
total = 0

top1_correct = 0
top5_correct = 0
total = 0

for seq in test_df['Sequence'][:500]:
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
    except:
        continue

print(f"Top-1 Accuracy: {top1_correct / total:.4f}")
print(f"Top-5 Accuracy: {top5_correct / total:.4f}")
