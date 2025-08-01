from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import DataCollatorWithPadding


def format_sequence(seq):
    """
    Formats a protein sequence by inserting spaces between each amino acid.

    Args:
        seq (str): A string representing a protein sequence (e.g., "MKWVTFISLL").

    Returns:
        str: A string with spaces between each amino acid (e.g., "M K W V T F I S L L").
    """
    # Strip any leading/trailing whitespace and insert spaces between each character
    return ' '.join(list(seq.strip()))


def get_spaced_sequence(train_df, test_df):
    """
    Applies formatting to protein sequences in the training and testing dataframes
    by adding spaces between amino acids. Adds a new column `spaced_sequence` to each dataframe.

    Args:
        train_df (pd.DataFrame): Training dataframe with a 'Sequence' column.
        test_df (pd.DataFrame): Testing dataframe with a 'Sequence' column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Modified training and testing dataframes
        with an additional 'spaced_sequence' column.
    """
    # Apply spacing to each protein sequence in the training set
    train_df['spaced_sequence'] = train_df['Sequence'].apply(format_sequence)

    # Apply spacing to each protein sequence in the testing set
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