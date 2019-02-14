"""
@author: Tobias Carryer
"""

import numpy as np
import math


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


def interleave(to_interleave, to_interleave2):
    """
    If to_interleave is [1,1,1] and to_interleave2 is [2,2,2,2,2] the output of this function would be [1,2,1,2,1,2,2,2]

    to_interleave comes first, that is, if the two arrays are the same length then to_interleave will make up the even
    numbered indices and to_interleave2 would make up the odd numbered indices.

    :param to_interleave: Array to interleave with to_interleave2
    :param to_interleave2: Array to interleave with to_interleave
    :return: The arrays interleaved with each other.
    """
    len_to_interleave = min(len(to_interleave), len(to_interleave2))
    interloven = [None] * (len_to_interleave * 2)  # Create array of the right length

    interloven[::2] = to_interleave[:len_to_interleave]
    interloven[1::2] = to_interleave2[:len_to_interleave]

    # Only the one with the values left will have anything left to append
    interloven += to_interleave[len_to_interleave:]
    interloven += to_interleave2[len_to_interleave:]

    return interloven


def get_subject_data(df, subject_number):
    """
    Subjects are identified by their subject_id, which is confocal_NUMBER
    :param df: Data frame of all subject data.
    :param subject_number: The value to fill in for NUMBER in confocal_NUMBER to get the subject id.
    :return: Data frame with all the data for the subject.
    """
    subject_data = df.loc[df['subject_id'] == "confocal_" + str(subject_number)]
    return subject_data


def make_dictionary(data, subject_numbers):
    """
    :param data: Array of DataFrames
    :param subject_numbers: Subject number for each DataFrame
    :return: Dictionary. Key: Subject number. Value: Data
    """
    return dict(zip(subject_numbers, data))


def get_i_for_split(subject_data, subject_number, min_date_for_subject, observation_size):
    """

    :param subject_data:
    :param subject_number:
    :param min_date_for_subject:
    :param observation_size:
    :return:
    """
    train_set_percentage = 0.6
    cv_set_percentage = 0.2
    test_set_percentage = 0.2

    n_for_training = math.ceil(len(subject_data) * train_set_percentage)
    n_for_cv = math.ceil(len(subject_data) * cv_set_percentage)

    if subject_number in min_date_for_subject:
        row_with_extreme = subject_data.loc[subject_data["time"] == min_date_for_subject[subject_number]]
        int_index_of_extreme_row = row_with_extreme.index.values.astype(int)[0]
        min_i = int_index_of_extreme_row + 1  # + 1 because the last index when splicing is exclusive

        if min_i > n_for_training:
            n_for_training = min_i

            percentage_used_for_training = n_for_training / len(subject_data)
            percentage_left = 1 - percentage_used_for_training
            percentage_for_cv = cv_set_percentage / (cv_set_percentage + test_set_percentage)

            n_for_cv = math.ceil(len(subject_data) * percentage_left * percentage_for_cv)

            if n_for_cv < observation_size:
                n_for_training = len(subject_data)
                n_for_cv = 0

        n_for_training = max(n_for_training, min_i)

    return n_for_training, n_for_cv

def get_data_split_up(df, labels, min_date_for_subject, observation_size):
    """
    Splits up the subjects into train, cv, and test sets.
    :param df: Data frame of all the subject data
    :param labels: Dictionary. Key: Integer. Values: Integer.
    :param min_date_for_subject: Dictionary. Key: Subject number of a subject with the most extreme value of a feature.
                                 Value: The time of the row with the most extreme value for the subject.
    :param observation_size: Integer
    :return: Three dictionaries (training, cv, testing). The keys are subject numbers and values are data frames.
    """
    pos_train_data = []
    pos_cv_data = []
    pos_test_data = []
    pos_subject_num = []

    neg_train_data = []
    neg_cv_data = []
    neg_test_data = []
    neg_subject_num = []

    for subject_number in get_subject_numbers(df):
        subject_data = get_subject_data(df, subject_number)

        n_for_training, n_for_cv = get_i_for_split(subject_data, subject_number, min_date_for_subject, observation_size)

        last_train_set_index = n_for_training
        last_cv_set_index = last_train_set_index + n_for_cv

        if labels[subject_number] == 1:
            pos_train_data.append(subject_data[:last_train_set_index])
            pos_cv_data.append(subject_data[last_train_set_index:last_cv_set_index])
            pos_test_data.append(subject_data[last_cv_set_index:])
            pos_subject_num.append(subject_number)
        else:
            neg_train_data.append(subject_data[:last_train_set_index])
            neg_cv_data.append(subject_data[last_train_set_index:last_cv_set_index])
            neg_test_data.append(subject_data[last_cv_set_index:])
            neg_subject_num.append(subject_number)

    # Interleave the data with 1 and 0 labels to make training more efficient. Ideally, the distribution between
    # the labels would be the same in each batch.
    train_data_list = interleave(pos_train_data, neg_train_data)
    cv_data_list = interleave(pos_cv_data, neg_cv_data)
    test_data_list = interleave(pos_test_data, neg_test_data)

    # The same interleave process is used as for interleaving the data so the subject numbers will correspond with
    # the values in the lists above
    subject_numbers = interleave(pos_subject_num, neg_subject_num)

    # Make a dictionary where the subject number is the key so it is easy to use it when iterating over all the data
    train_data = make_dictionary(train_data_list, subject_numbers)
    cv_data = make_dictionary(cv_data_list, subject_numbers)
    test_data = make_dictionary(test_data_list, subject_numbers)

    return train_data, cv_data, test_data
