"""
@author: Tobias Carryer
"""

import numpy as np
import math
import random


test_set = [27, 32, 35, 39, 43, 52]


def get_subject_numbers(df):
    """
    Get the subject numbers in the data frame.
    :param df: Data frame to get subject numbers from.
    :return: An array of the subject numbers.
    """
    subject_ids = df["subject_id"].unique()
    subject_numbers = []
    for subject_id in subject_ids:
        subject_numbers.append(int(subject_id.lstrip("confocal_")))
    return subject_numbers


def get_subject_data(df, subject_number):
    """
    Subjects are identified by their subject_id, which is confocal_NUMBER
    :param df: Data frame of all subject data.
    :param subject_number: The value to fill in for NUMBER in confocal_NUMBER to get the subject id.
    :return: Data frame with all the data for the subject.
    """
    subject_data = df.loc[df['subject_id'] == "confocal_" + str(subject_number)]
    return subject_data


def number_of_observations_per_set(observations, subject_number, min_date_for_subject):
    """
    :param observations: Array of DataFrames
    :param subject_number: Subject number to create observations for
    :param min_date_for_subject: Dictionary with the minimum date that must be included for subjects with extreme values
    :return: Integer (number of entries to go in training), Integer (number of entries to go in the CV set)
    """
    train_set_percentage = 0.8  # CV set is last 20% of time series data, test set is completely separate

    n_for_training = math.ceil(len(observations) * train_set_percentage)

    # Include the last observation with an extreme value if applicable
    if subject_number in min_date_for_subject:
        min_i = 0
        for i, observation in enumerate(observations):
            if min_date_for_subject[subject_number] in observation["time"]:
                min_i = i

        n_for_training = max(n_for_training, min_i + 1)  # +1 to account for 0 indexing

    return n_for_training


def create_observations(df, subject_label, observation_size):
    """
    Creates an array of observations and the corresponding labels. The observations are a rolling window of the data
    frame, that is, a subset of the data frame is copied for every observation.
    :param df: DataFrame to split up into observations
    :param subject_label: 1 or 0, to be repeated as each observation's label
    :param observation_size: The size of each observation
    :return: Array (observations), Array (labels)
    """
    observations = []
    labels = []
    for i in range(len(df)-observation_size):
        df_observation = df.iloc[i:i + observation_size]

        # GASF-GADF-MTF compound images cannot contain NaNs so we filter them out those observations
        has_nan = np.isnan(df_observation["BtO2"]).any() or np.isnan(df_observation["HR"]).any() or \
                  np.isnan(df_observation["SpO2"]).any() or np.isnan(df_observation["artMAP"]).any()

        if not has_nan:
            observations.append(df_observation)
            labels.append(subject_label)
    return observations, labels


def shuffle_together(data, labels):
    """
    Shuffles two arrays in sync so labels continue corresponding to the right data.
    The shuffling is done with a seed so it will produce the same results every time.
    :param data: Array
    :param labels: Array
    :return: Returns a tuple, the data and the labels.
    """
    tmp = list(zip(data, labels))
    random.Random(42).shuffle(tmp)
    data, labels = zip(*tmp)
    return list(data), list(labels)


def get_data_split_up(df, labels, min_date_for_subject, observation_size):
    """
    Splits up the subjects into train, cv, and test sets.
    :param df: Data frame of all the subject data
    :param labels: Dictionary. Key: Integer, subject number. Values: Integer.
    :param min_date_for_subject: Dictionary. Key: Integer, subject number of a subject with the most extreme value of a
                                feature. Value: The time of the row with the most extreme value for the subject.
    :param observation_size: Integer
    :return: Four tuples. Each tuple is (data, labels). The first two tuples contains 1D arrays. The next two tuples
             contain 2D arrays where each subarray is a subject's data or labels.
    """
    train_data = []
    cv_data = []  # 1D array for cross validation after each epoch
    cv_data_for_evaluating = {}  # 2D array to get average accuracy per subject using evaluate_generator
    test_data = {}

    train_lbls = []
    cv_lbls = []
    cv_lbls_for_evaluating = {}
    test_lbls = {}

    for subject_number in get_subject_numbers(df):
        subject_data = get_subject_data(df, subject_number)
        observations, lbls = create_observations(subject_data, labels[subject_number], observation_size)

        # Test set subjects are kept entirely separate
        if subject_number in test_set:
            test_data[subject_number] = observations
            test_lbls[subject_number] = lbls
            continue

        # Split is usually done by a percentage of data, but it also depends on whether or not the subject has
        # an extreme value that determines how the MinMaxScalers are trained. In that case, the extreme value
        # has to go in the train set to prevent information leakage.
        n_for_training = number_of_observations_per_set(observations, subject_number, min_date_for_subject)

        train_data += observations[:n_for_training]
        cv_data += observations[n_for_training:]  # Add the array elements
        cv_data_for_evaluating[subject_number] = observations[n_for_training:]  # Add the array itself

        train_lbls += lbls[:n_for_training]
        cv_lbls += lbls[n_for_training:]
        cv_lbls_for_evaluating[subject_number] = lbls[n_for_training:]

    # Necessary for the model to learn. Otherwise, each batch only contains one subject.
    train_data, train_lbls = shuffle_together(train_data, train_lbls)

    return (train_data, train_lbls), (cv_data, cv_lbls),\
           (cv_data_for_evaluating, cv_lbls_for_evaluating), (test_data, test_lbls)
