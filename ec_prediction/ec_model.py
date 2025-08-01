import json
from dataset import load_dataset
from data_prepcess import get_spaced_sequence, load_ec_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModelForSequenceClassification
from transformers import EsmTokenizer, EsmForMaskedLM
from evaluation import compute_metrics

class ECModel:
    """
        A class to fine-tune a masked language model (MLM) such as RoBERTa /
        ModernBERT / Distilbert / ESM / on protein sequence data for enzyme
        classification (EC) prediction.
    """
    def __init__(self, args):
        self.data_collator = None
        self.tokenized_dataset_test = None
        self.tokenized_dataset_train = None
        self.num_labels = None
        self.model_name = args.model
        self.tokenize_name = args.tokenizer
        if 'esm' in self.model_name:
            self.tokenizer = EsmTokenizer.from_pretrained(args.tokenizer, do_lower_case=False)
            self.model = EsmForMaskedLM.from_pretrained(args.model)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenize_name, do_lower_case=False)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.mlm_probability = args.mlm_probability
        self.output_directory = args.output_dir
        self.train_data_path = args.train_csv_path
        self.test_data_path = args.test_csv_path
        self.epochs = args.epochs
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.save_strategy = args.save_strategy

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
        df_train, self.label_encoder = load_ec_dataset(df_train, self.tokenizer)
        df_test, _ = load_ec_dataset(df_test, self.tokenizer)

        self.tokenized_dataset_train = df_train.map(self.tokenize_fn, batched=True)
        self.tokenized_dataset_test = df_test.map(self.tokenize_fn, batched=True)


    def call_trainer(self):

            self.num_labels = len(self.label_encoder.classes_)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )

            training_args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                eval_strategy="epoch",
                save_strategy="epoch",
                logging_dir="./logs",
                learning_rate=2e-5,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
            )

            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Trainer
            # Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=self.tokenized_dataset_train,
                eval_dataset=self.tokenized_dataset_test,
                compute_metrics=compute_metrics
            )

            # Train
            trainer.train()

            # Evaluate
            eval_results = trainer.evaluate()
            print("Evaluation Results:", eval_results)

