import pandas as pd

def load_dataset(train_data_path, test_data_path):
    """
    :param args:
    :return:
    """
    # Step 1: Load your train data
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_df = train_df[train_df['Sequence'].str.len() > 20]  # remove short sequences
    test_df = test_df[test_df['Sequence'].str.len() > 20]  # remove short sequences

    train_df = train_df.iloc[0:400,:]
    test_df = test_df.iloc[0:100,:]

    return train_df, test_df