"""
@author: Shreyansh Anand
@:param df - it takes in a dataframe and causes 10% of the values to become NaNs
"""

import random


def add_noise(df):
    # We add 0s randomly into the data in order to make
    # noise which can then be used to train the model on
    # NaN values.

    matrix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    # matrix = dataframe, used to randomly find rows and cols

    for row, col in random.sample(matrix, int(round(.1 * len(matrix)))):
        df.iat[row, col] = 0
        # take the df and go into it at random rows and col and for 0.1% (10%)
        # of them, make them into 0.
    return df
