from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import DataCollatorWithPadding



# Step 4: Format sequences with spaces between amino acids
def format_sequence(seq):
    """
    :param seq:
    :return:
    """
    return ' '.join(list(seq.strip()))

def get_spaced_sequence(train_df, test_df):
    """
    :param train_df:
    :param test_df:
    :return:
    """
    train_df['spaced_sequence'] = train_df['Sequence'].apply(format_sequence)
    test_df['spaced_sequence'] = test_df['Sequence'].apply(format_sequence)


    return train_df, test_df

def load_ec_dataset(df, tokenizer, max_length=512):
    """
    Prepares a HuggingFace Dataset from a pandas DataFrame by:
      1. Encoding the EC (enzyme classification) labels into integers.
      2. Tokenizing the spaced protein sequences.

    Args:
        df (pd.DataFrame): DataFrame with 'spaced_sequence' and 'EC' columns.
        tokenizer (PreTrainedTokenizer): A HuggingFace tokenizer to tokenize the sequences.
        max_length (int): Maximum length for the tokenized sequences (default: 512).

    Returns:
        tuple:
            - dataset (Dataset): Tokenized HuggingFace Dataset with 'input_ids', 'attention_mask', and 'label'.
            - label_encoder (LabelEncoder): Fitted encoder to convert model predictions back to EC labels.
    """
    # Encode EC numbers into integer labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['EC'])

    # Tokenize sequences
    def tokenize_function(examples):
        return tokenizer(
            examples["spaced_sequence"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )

    dataset = Dataset.from_pandas(df[['spaced_sequence', 'label']])
    dataset = dataset.map(tokenize_function, batched=True)

    return dataset, label_encoder