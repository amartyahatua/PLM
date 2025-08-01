from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import optuna

# 1. Load ModernBERT tokenizer and model
model_name = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def model_init():
    return AutoModelForMaskedLM.from_pretrained(model_name)

# 2. Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 3. Create DataCollator for MLM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15,
)

# 4. Training arguments (base, will be overridden by hyperparameter search)
training_args = TrainingArguments(
    output_dir="./modernbert-mlm",
    eval_strategy="no",
    save_strategy="no",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 5. Trainer setup
trainer = Trainer(
    model_init=model_init,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 6. Define search space for Optuna
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
    }

# 7. Run hyperparameter search
best_trial = trainer.hyperparameter_search(
    direction="minimize",
    backend="optuna",
    hp_space=hp_space,
    n_trials=10
)

print("Best hyperparameters:", best_trial.hyperparameters)

# 8. Retrain final model with best hyperparameters
best_args = best_trial.hyperparameters

final_training_args = TrainingArguments(
    output_dir="./modernbert-mlm-best",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_args["learning_rate"],
    per_device_train_batch_size=best_args["per_device_train_batch_size"],
    num_train_epochs=best_args["num_train_epochs"],
    weight_decay=best_args["weight_decay"],
)

final_trainer = Trainer(
    model_init=model_init,
    args=final_training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

final_trainer.train()

# 9. Save the fine-tuned model
final_trainer.save_model("./modernbert-mlm-finetuned")
tokenizer.save_pretrained("./modernbert-mlm-finetuned")

print("Fine-tuning completed. Model saved.")
