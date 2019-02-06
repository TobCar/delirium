"""
@author: Tobias Carryer
"""

import numpy as np


def vote(predicted_labels, voting_threshold=0.5):
    """
    :param predicted_labels: Array of labels predicted
    :param voting_threshold: The label predicted this percent of the time, or more, will be chosen.
    :return: The label predicted the most in predicted_labels
    """
    percent_true = np.count_nonzero(predicted_labels) / len(predicted_labels)
    if percent_true >= voting_threshold:
        return 1
    else:
        return 0