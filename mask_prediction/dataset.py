import pandas as pd

def load_dataset(train_data_path, test_data_path):
    """
    Loads training and testing protein sequence data from CSV files,
    applies basic filtering, and returns processed DataFrames.

    Parameters:
        train_data_path (str): Path to the CSV file containing training data.
                               Must include a 'Sequence' column.
        test_data_path (str): Path to the CSV file containing testing data.
                              Must include a 'Sequence' column.

    Returns:
        tuple: (train_df, test_df)
            - train_df (pandas.DataFrame): Filtered training dataset with sequences >20 amino acids,
              limited to the first 1000 rows.
            - test_df (pandas.DataFrame): Filtered testing dataset with sequences >20 amino acids,
              limited to the first 500 rows.

    Processing Steps:
        1. Reads training and testing CSV files into Pandas DataFrames.
        2. Filters out rows where the 'Sequence' column has 20 or fewer characters
           (removes very short protein sequences).
        3. Keeps only the first 1000 rows of training data and the first 500 rows of testing data
           for faster experimentation.
    """
    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    train_df = train_df[train_df['Sequence'].str.len() > 20]  # remove short sequences
    test_df = test_df[test_df['Sequence'].str.len() > 20]  # remove short sequences

    # Get a small dataset
    # train_df = train_df.iloc[0:1000,:]
    # test_df = test_df.iloc[0:500,:]

    return train_df, test_df