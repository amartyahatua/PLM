from datasets import Dataset


# Step 4: Format sequences with spaces between amino acids
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
    test_df['spaced_sequence'] = test_df['Sequence'].apply(format_sequence)

    # Step 5: Convert to HuggingFace Dataset
    dataset_train = Dataset.from_pandas(train_df[['spaced_sequence']])
    dataset_test = Dataset.from_pandas(test_df[['spaced_sequence']])

    return dataset_train, dataset_test