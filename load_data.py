"""
@author: Tobias Carryer
"""

import pandas as pd


def load_data_file(path_to_file, timestamp_column_name="time"):
    """
    :param path_to_file: The path to the file to load.
    :return: The data frame of the data set.
             The column names are modified to be lowercase.
             The timestamps are used as the data frame indices.
    """

    df = pd.read_csv(path_to_file)
    df.columns = df.columns.str.lower()  # Make column names lowercase.
    df.set_index(timestamp_column_name, inplace=True)   # Make the timestamp the index.
    return df
