"""
@author Tobias Carryer (fill_in_data) and Max Berkowitz (fill_in_df)
"""


def fill_in_data(data):
    """
    :param data: Dictionary. Key: Integer. Value: Data frame.
    :return: Dictionary with updated values.
    """
    for key, df in data.items():
        data[key] = fill_in_df(df)
    return data


def fill_in_df(df):
    column_names = df.columns[2:]  # Skip time and subject ID columns
    for column in column_names:
        df[column] = df[column].rolling(window=200, min_periods=1).mean()
    return df