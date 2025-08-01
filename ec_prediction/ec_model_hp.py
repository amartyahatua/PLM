import json
from transformers import EsmTokenizer, EsmForMaskedLM
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer, TrainingArguments
from dataset import load_dataset
from data_prepcess import get_spaced_sequence, load_ec_dataset
from hp_finetune import hp_space

class MaskModelHP:
    """
        A class to fine-tune a masked language model (MLM) such as RoBERTa /
        ModernBERT / Distilbert / ESM / on protein sequence data for enzyme
        classification (EC) prediction.
    """
    def __init__(self, args):
        self.data_collator = None
        self.tokenized_dataset_test = None
        self.tokenized_dataset_train = None
        self.model_name = args.model
        if 'esm' in self.model_name:
            self.tokenizer = EsmTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
            self.model = EsmForMaskedLM.from_pretrained(args.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
            self.model = AutoModelForMaskedLM.from_pretrained(args.model)
        self.mlm_probability = args.mlm_probability
        self.output_directory = args.output_dir
        self.train_data_path = args.train_csv_path
        self.test_data_path = args.test_csv_path
        self.epochs = args.epochs
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.save_strategy = args.save_strategy
        self.tokenize_name = args.tokenizer
        self.eval_strategy = args.eval_strategy
        self.logging_steps = args.logging_steps

    def tokenize_fn(self, examples):
        """
        Tokenizes a batch of protein sequences with spaces between amino acids.

        Args:
            examples (dict): Dictionary containing the key 'spaced_sequence',
                             a list of space-separated protein sequences.

        Returns:
            dict: Tokenized representation of the input sequences with padding and truncation.
        """
        return self.tokenizer(
            examples['spaced_sequence'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    def get_dataset(self):
        """
        Loads, preprocesses, and tokenizes the training and testing datasets.

        Steps:
            1. Loads the raw datasets from CSV files using `load_dataset`.
            2. Converts sequences into space-separated amino acid format using `get_spaced_sequence`.
            3. Tokenizes the datasets using `tokenize_fn` for model training and evaluation.

        Returns:
            None: Updates `tokenized_dataset_train` and `tokenized_dataset_test` attributes.
        """
        df_train, df_test = load_dataset(self.train_data_path, self.test_data_path)
        df_train, df_test = get_spaced_sequence(df_train, df_test)
        self.tokenized_dataset_train = df_train.map(self.tokenize_fn, batched=True)
        self.tokenized_dataset_test = df_test .map(self.tokenize_fn, batched=True)

        self.tokenized_dataset_train, self.label_encoder = load_ec_dataset(self.tokenized_dataset_train, self.tokenizer)
        self.tokenized_dataset_test, _ = load_ec_dataset(self.tokenized_dataset_test, self.tokenizer)

    def model_init(self):
        """
        Returns the model instance associated with this MaskModel object.

        This method is commonly used when performing hyperparameter tuning with
        Hugging Face's Trainer API, as it allows the Trainer to re-initialize
        a fresh copy of the model for each trial.

        Returns:
            transformers.PreTrainedModel: The masked language model (MLM) being fine-tuned.
        """
        return self.model

    def call_trainer(self):
        """
        Performs masked language model (MLM) fine-tuning with hyperparameter optimization
        using the Hugging Face `Trainer` API.

        Steps:
            1. **Create data collator:**
               Initializes `DataCollatorForLanguageModeling`, which dynamically applies
               random masking to tokens during training according to `mlm_probability`.

            2. **Initial training arguments:**
               Sets up `TrainingArguments` including output directory, number of epochs,
               batch size, save/evaluation strategies, and logging configuration.

            3. **Initial Trainer and hyperparameter search:**
               Creates a `Trainer` instance and runs hyperparameter optimization with `trainer.hyperparameter_search()`.
               Uses Optuna (or another supported backend) to minimize evaluation loss by tuning
               learning rate, weight decay, batch size, and number of epochs.

            4. **Save best hyperparameters:**
               Saves the best hyperparameters found during the search to a JSON file (`<model_name>_hp.json`).

            5. **Recreate training arguments with optimal hyperparameters:**
               Builds a new `TrainingArguments` object using the best settings for
               learning rate, batch size, number of epochs, and weight decay.

            6. **Final training:**
               Creates a new `Trainer` instance with the best hyperparameters and trains
               the model on the tokenized training dataset.

            7. **Evaluation:**
               Evaluates the fine-tuned model on the tokenized test dataset and prints the final evaluation loss.

            8. **Model saving:**
               Saves the final fine-tuned model and tokenizer to directories named
               `finetuned_protein_SwissprotDatasets_BalancedSwissprot` under the model and tokenizer names.

        Notes:
            - `trainer.hyperparameter_search()` requires a `hp_space` function
              defining the search space for the hyperparameters.
            - The initial `trainer` is used for hyperparameter search, and `final_trainer`
              is used for the final training with optimal parameters.
            - JSON serialization stores the full best trial details for reproducibility.
        """

        # Step 1: Create data collator
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability
        )

        # Step 2: Set training arguments
        training_args = TrainingArguments(
            output_dir=self.output_directory,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            save_strategy=self.save_strategy,
            eval_strategy=self.eval_strategy,
            logging_steps=self.logging_steps,
            logging_dir=f"../{self.model_name}/logs",
        )

        # Step 3: Set trainer
        trainer = Trainer(
            model_init=self.model_init,
            args=training_args,
            train_dataset=self.tokenized_dataset_train,
            eval_dataset=self.tokenized_dataset_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        # Step 4: Get best hyperparameters
        best_trial = trainer.hyperparameter_search(
            direction="minimize",  # minimize loss
            backend="optuna",  # or "ray", "sigopt"
            hp_space=hp_space,
            n_trials=10  # number of trials
        )

        # Step 5: Save the best hyperparameters
        print("Best trial:", best_trial)

        best_hp_path = f"{self.model_name}_hp.json"
        with open(best_hp_path, "w") as json_file:
            json.dump(best_trial, json_file, indent=4)

        # Step 6: Set best training arguments
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

        # Step 7: Set the trainer
        final_trainer = Trainer(
            model=self.model,
            args=best_training_args,
            train_dataset=self.tokenized_dataset_train,
            eval_dataset=self.tokenized_dataset_test,
            tokenizer=self.tokenizer,
            data_collator=self.data_collator
        )

        # Step 8: Train
        final_trainer.train()

        # Step 9: Final evaluation
        eval_results = final_trainer.evaluate()
        print(f"\n✅ Final eval loss: {eval_results['eval_loss']:.4f}")

        # Step 10: Save model
        final_trainer.save_model(f"{self.model_name}/EC_Prediction_protein_SwissprotDatasets_BalancedSwissprot")
        self.tokenizer.save_pretrained(f"{self.tokenize_name}/EC_Prediction_protein_SwissprotDatasets_BalancedSwissprot")
