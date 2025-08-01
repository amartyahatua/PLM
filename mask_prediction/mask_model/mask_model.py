# Step 1: Import Libraries
import json
from mask_prediction.roberta.dataset import load_dataset
from mask_prediction.roberta.data_prepcess import get_spaced_sequence
from hp_finetune import hp_space
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer, TrainingArguments


class MaskModel:
    def __init__(self, args):
        self.data_collator = None
        self.tokenized_dataset_test = None
        self.tokenized_dataset_train = None

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
        self.mlm_probability = args.mlm_probability
        self.output_directory = args.output_dir
        self.train_data_path = args.train_csv_path
        self.test_data_path = args.test_csv_path
        self.epochs = args.epochs
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.save_strategy = args.save_strategy
        self.model_name = args.model
        self.tokenize_name = args.tokenizer
        self.eval_strategy = args.eval_strategy
        self.logging_steps = args.logging_steps
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

    def tokenize_fn(self, examples):
        """
        :param examples:
        :return:
        """
        return self.tokenizer(
            examples['spaced_sequence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    def get_dataset(self):
        """
        :return:
        """
        df_train, df_test = load_dataset(self.train_data_path, self.test_data_path)
        df_train, df_test = get_spaced_sequence(df_train, df_test)
        self.tokenized_dataset_train = df_train.map(self.tokenize_fn, batched=True)
        self.tokenized_dataset_test = df_test .map(self.tokenize_fn, batched=True)

    def model_init(self):
        return self.model

    def call_trainer(self):
        """
        :return:
        """
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability
        )

        training_args = TrainingArguments(
            output_dir=self.output_directory,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            save_strategy=self.save_strategy,
            eval_strategy=self.eval_strategy,
            logging_steps=self.logging_steps,
            logging_dir=f"../{self.model_name}/logs"
        )

        # Step 10: Trainer
        trainer = Trainer(
            model=self.model_init,
            args=training_args,
            train_dataset=self.tokenized_dataset_train,
            eval_dataset=self.tokenized_dataset_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        best_trial = trainer.hyperparameter_search(
            direction="minimize",  # minimize loss
            backend="optuna",  # or "ray", "sigopt"
            hp_space=hp_space,
            n_trials=10  # number of trials
        )

        # Save the best hyperparameters
        print("Best trial:", best_trial)

        best_hp_path = f"{self.model_name}_hp.json"
        with open(best_hp_path, "w") as json_file:
            json.dump(best_trial, json_file, indent=4)

        best_training_args = TrainingArguments(
            output_dir=self.output_directory,
            num_train_epochs=best_trial["num_train_epochs"],
            weight_decay=best_trial["weight_decay"],
            per_device_train_batch_size=best_trial['per_device_train_batch_size'],
            save_strategy=self.save_strategy,
            eval_strategy=self.eval_strategy,
            logging_steps=self.logging_steps,
            logging_dir=f"../{self.model_name}/logs",
            learning_rate=best_trial["learning_rate"],
        )

        # Step 10: Trainer
        final_trainer = Trainer(
            model=self.model,
            args=best_training_args,
            train_dataset=self.tokenized_dataset_train,
            eval_dataset=self.tokenized_dataset_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        # Step 11: Train
        final_trainer.train()

        # Step 12: Final evaluation
        eval_results = trainer.evaluate()
        print(f"\n✅ Final eval loss: {eval_results['eval_loss']:.4f}")

        # Step 13: Save model
        trainer.save_model(f"{self.model_name}/finetuned_protein_SwissprotDatasets_BalancedSwissprot")
        self.tokenizer.save_pretrained(f"{self.tokenize_name}/finetuned_protein_SwissprotDatasets_BalancedSwissprot")

