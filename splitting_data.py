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
        subject_numbers.append(subject_id.lstrip("confocal_"))
    return subject_numbers


def split_subject_numbers(subject_numbers, must_go_in_training):
    """
    :param subject_numbers: Subject numbers for the subjects in df.
    :param must_go_in_training: List of subject numbers that must go in the training set
    :return: Three arrays of subject numbers meant to be used for the train set, the cv set, and the test set.
    """
    train_set_percentage = 0.6
    cv_set_percentage = 0.2

    n_for_training = math.ceil(len(subject_numbers)*train_set_percentage)
    n_for_cv = math.ceil(len(subject_numbers) * cv_set_percentage)

    # Start by assigning the mandatory numbers
    train_set_subject_numbers = []
    for must_go in must_go_in_training:
        if must_go in subject_numbers:
            train_set_subject_numbers.append(must_go)
            subject_numbers.remove(must_go)  # Prevent adding it twice
            n_for_training -= 1

    # Assign the subjects to the sets at random, following the distribution from the percentages above
    shuffled_subject_numbers = np.random.permutation(subject_numbers).tolist()
    last_train_set_index = n_for_training
    last_cv_set_index = last_train_set_index + n_for_cv

    train_set_subject_numbers += shuffled_subject_numbers[:last_train_set_index]
    cv_set_subject_numbers = shuffled_subject_numbers[last_train_set_index:last_cv_set_index]
    test_set_subject_numbers = shuffled_subject_numbers[last_cv_set_index:]

    return train_set_subject_numbers, cv_set_subject_numbers, test_set_subject_numbers


def assign_subject_numbers_to_splits(delirious_subject_numbers, non_delirious_subject_numbers, must_go_in_training):
    """
    :param delirious_subject_numbers: Array of subject numbers of all patients with delirium.
    :param non_delirious_subject_numbers: Array of subject numbers of all patients without delirium.
    :param must_go_in_training: List of subject numbers that must go in the training set
    :return: Array of the subject numbers that belong in each of the train, cv, and test sets.
             The distribution of delirious and non-delirious patients is the same in all sets.
    """
    train_subject_nums, cv_subject_nums, test_subject_nums = split_subject_numbers(delirious_subject_numbers,
                                                                                   must_go_in_training)
    train_subject_nums2, cv_subject_nums2, test_subject_nums2 = split_subject_numbers(non_delirious_subject_numbers,
                                                                                      must_go_in_training)

    # Interleaves to prevent getting stuck in local optima from only seeing one type of subject for half the epoch.
    train_subject_nums = interleave(train_subject_nums, train_subject_nums2)
    cv_subject_nums = interleave(cv_subject_nums, cv_subject_nums2)
    test_subject_nums = interleave(test_subject_nums, test_subject_nums2)

    return train_subject_nums, cv_subject_nums, test_subject_nums


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


def get_data_for_splits(df, train_subject_nums, cv_subject_nums, test_subject_nums):
    """
    :param train_subject_nums: Array of the subjects that go in the train set.
    :param cv_subject_nums: Array of the subjects that go in the cv set.
    :param test_subject_nums: Array of the number of subjects that go in the test set.
    :return: Dictionaries of data frames to use as the train, cv, and test set.
             The key for the dictionary is the patient number, an integer.
             Each subject is its own data frame because time series data is independent of other subjects.
    """
    train_data = {}
    for subject_number in train_subject_nums:
        subject_data = get_subject_data(df, subject_number)
        subject_data = subject_data.reset_index(drop=True)
        train_data[subject_number] = subject_data

    cv_data = {}
    for subject_number in cv_subject_nums:
        subject_data = get_subject_data(df, subject_number)
        subject_data = subject_data.reset_index(drop=True)
        cv_data[subject_number] = subject_data

    test_data = {}
    for subject_number in test_subject_nums:
        subject_data = get_subject_data(df, subject_number)
        subject_data = subject_data.reset_index(drop=True)
        test_data[subject_number] = subject_data

    return train_data, cv_data, test_data


def get_data_split_up(all_subject_data, labels, must_go_in_training):
    """
    Splits up the subjects into train, cv, and test sets.
    :param all_subject_data: Data frame
    :param labels: Dictionary. Key: Integer. Values: Integer.
    :param must_go_in_training: List of subject numbers that must go in the training set
    :return: Three dictionaries (training, cv, testing). The keys are subject numbers and values are data frames.
    """
    # Identify what subjects have delirium
    positive_subject_numbers = []
    negative_subject_numbers = []
    for subject_number, label in labels.items():
        if label == 1:
            positive_subject_numbers.append(subject_number)
        elif label == 0:
            negative_subject_numbers.append(subject_number)
        else:
            raise ValueError("Expected 1 or 0. Received label: " + str(label))

    # Split up the patients into train/cv/test sets as subject numbers first, then get the respective data
    train_subject_nums, cv_subject_nums, test_subject_nums = assign_subject_numbers_to_splits(positive_subject_numbers,
                                                                                              negative_subject_numbers,
                                                                                              must_go_in_training)
    train_data, cv_data, test_data = get_data_for_splits(all_subject_data,
                                                         train_subject_nums,
                                                         cv_subject_nums,
                                                         test_subject_nums)

    return train_data, cv_data, test_data
