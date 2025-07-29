from datasets import Dataset


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

    # Step 5: Convert to HuggingFace Dataset
    dataset_train = Dataset.from_pandas(train_df[['spaced_sequence']])
    dataset_test = Dataset.from_pandas(test_df[['spaced_sequence']])

    return dataset_train, dataset_test