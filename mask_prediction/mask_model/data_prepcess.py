from datasets import Dataset

def format_sequence(seq):
    """
    Takes a protein sequence string (a continuous string of amino acid single-letter codes)
    and inserts spaces between each character.

    Parameters:
        seq (str): A string representing a protein sequence, e.g. "MKTIIALSYIFCLVFA".

    Returns:
        str: A new string where each amino acid is separated by a space, e.g. "M K T I I A L S Y I F C L V F A".

    Purpose:
        Many tokenizers for protein language models expect input where each amino acid
        is separated by spaces to correctly identify tokens.
    """
    return ' '.join(list(seq.strip()))

def get_spaced_sequence(train_df, test_df):
    """
    Processes training and testing DataFrames by formatting their protein sequences
    and converting them into Hugging Face Datasets.

    Parameters:
        train_df (pandas.DataFrame): DataFrame containing a 'Sequence' column with protein sequences for training.
        test_df (pandas.DataFrame): DataFrame containing a 'Sequence' column with protein sequences for testing.

    Returns:
        tuple: (dataset_train, dataset_test)
            - dataset_train: Hugging Face Dataset with a 'spaced_sequence' column for training.
            - dataset_test: Hugging Face Dataset with a 'spaced_sequence' column for testing.

    Steps:
        1. Applies `format_sequence` to the 'Sequence' column of both train_df and test_df,
           creating a new column 'spaced_sequence' with space-separated amino acids.
        2. Converts these processed DataFrames into Hugging Face `Dataset` objects for use
           in model training and evaluation.
    """
    train_df['spaced_sequence'] = train_df['Sequence'].apply(format_sequence)
    test_df['spaced_sequence'] = test_df['Sequence'].apply(format_sequence)

    # Step 5: Convert to HuggingFace Dataset
    dataset_train = Dataset.from_pandas(train_df[['spaced_sequence']])
    dataset_test = Dataset.from_pandas(test_df[['spaced_sequence']])

    return dataset_train, dataset_test