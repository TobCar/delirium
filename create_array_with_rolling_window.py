"""
@author: Tobias Carryer
"""

import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided


def create_array_with_rolling_window(series, window_size=128):
    """
    Takes the first window_size values and puts them in the array. Then, it moves the window over by one row and repeats
    the process. This creates an array of observations of size window_size.

    Example: Series is represented as this array: [1, 2, 3, 4, 5]. The window_size equals 2.
             In this example the function would return [[1,2], [2,3], [3,4], [4,5]].

    :param series: Series to turn into 2D array of observations.
    :param window_size: Number of rows to put in one entry of the array.
    :return: A read-only 2D numpy array. Will be of shape (0, window_size) if the series is shorter than window_size.
    """
    if len(series) < window_size:
        return np.empty((0, window_size))

    values = series.values
    stride = values.strides[0]
    number_of_values = values.shape[0]

    # This is the same as calculating the resulting dimension after a convolution.
    number_of_observations = number_of_values - window_size + 1

    # writeable=False is to protect against future problems if this function is modified
    # numpy discusses the risk of using as_strided() in in their documentation.
    out = as_strided(values, shape=(number_of_observations, window_size), strides=(stride, stride),
                     writeable=False)

    return out


if __name__ == "__main__":
    test_df = pd.DataFrame({"close": [10, 11, 12, 13],
                            "open": [9, 10, 11, 12],
                            "date": ["2016", "2017", "2018", "2019"]})
    test_df.set_index("date", inplace=True)
    test_series = test_df["close"]

    print("Testing with the series:")
    print(test_series)

    observations = create_array_with_rolling_window(test_series, window_size=2)

    print("Function created the following array of observations:")
    print(observations)

    assert(len(observations) == 3)
    assert(observations[0][0] == 10 and observations[0][1] == 11)
    assert(observations[1][0] == 11 and observations[1][1] == 12)
    assert(observations[2][0] == 12 and observations[2][1] == 13)
