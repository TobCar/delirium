"""
@author: Tobias Carryer
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def validate_min_max_scalers(scalers, cv_data, test_data):
    """
    GASF-GADF-MTF compound images only work if the values are normalized. If the CV or test set has more extreme values
    than the training set then normalization will break the compound images in those sets. It is not ideal but the
    subjects must be assigned to the training set such that the MinMaxScalers created for it span all possible
    values the CV and test set will take on. This function asserts the subjects have been assigned to the training
    set accordingly.

    :param scalers: Dictionary. Key: String, feature to normalize. Value: MinMaxScaler.
    :param cv_data: Dictionary. Key: Integer, subject number. Value: Data frame of the data from the subject.
    :param test_data: Dictionary. Key: Integer, subject number. Value: Data frame of the data from the subject.
    """
    cv_scalers = create_min_max_scalers(cv_data)
    test_scalers = create_min_max_scalers(test_data)

    assert (scalers["HR"].data_max_ >= cv_scalers["HR"].data_max_)
    assert (scalers["HR"].data_max_ >= test_scalers["HR"].data_max_)
    assert (scalers["HR"].data_min_ <= cv_scalers["HR"].data_min_)
    assert (scalers["HR"].data_min_ <= test_scalers["HR"].data_min_)
    assert (scalers["BtO2"].data_max_ >= cv_scalers["BtO2"].data_max_)
    assert (scalers["BtO2"].data_max_ >= test_scalers["BtO2"].data_max_)
    assert (scalers["BtO2"].data_min_ <= cv_scalers["BtO2"].data_min_)
    assert (scalers["BtO2"].data_min_ <= test_scalers["BtO2"].data_min_)
    assert (scalers["SpO2"].data_max_ >= cv_scalers["SpO2"].data_max_)
    assert (scalers["SpO2"].data_max_ >= test_scalers["SpO2"].data_max_)
    assert (scalers["SpO2"].data_min_ <= cv_scalers["SpO2"].data_min_)
    assert (scalers["SpO2"].data_min_ <= test_scalers["SpO2"].data_min_)
    assert (scalers["artMAP"].data_max_ >= cv_scalers["artMAP"].data_max_)
    assert (scalers["artMAP"].data_max_ >= test_scalers["artMAP"].data_max_)
    assert (scalers["artMAP"].data_min_ <= cv_scalers["artMAP"].data_min_)
    assert (scalers["artMAP"].data_min_ <= test_scalers["artMAP"].data_min_)


def create_min_max_scalers(train_data):
    """
    Creates the MinMaxScalers necessary to normalize the data.
    :param train_data: Dictionary. Key: Integer, subject number. Value: Data frame of the data from the subject.
    :return: Dictionary. Key: String, feature to normalize. Value: MinMaxScaler that can normalize the feature.
    """
    # We could create one scaler for all the features but this approach makes it easier to use the scalers outside
    # of this project if someone needs them.
    all_train_btO2_data = np.array([])
    all_train_hr_data = np.array([])
    all_train_spO2_data = np.array([])
    all_train_artmap_data = np.array([])

    # Will normalize all data together regardless of what patient contributed it.
    for data in train_data.values():
        all_train_btO2_data = np.append(all_train_btO2_data, data["BtO2"].tolist())
        all_train_hr_data = np.append(all_train_hr_data, data["HR"].tolist())
        all_train_spO2_data = np.append(all_train_spO2_data, data["SpO2"].tolist())
        all_train_artmap_data = np.append(all_train_artmap_data, data["artMAP"].tolist())

    btO2_scaler = MinMaxScaler()
    hr_scaler = MinMaxScaler()
    spO2_scaler = MinMaxScaler()
    artmap_scaler = MinMaxScaler()

    btO2_scaler.fit(all_train_btO2_data.reshape(-1, 1))
    hr_scaler.fit(all_train_hr_data.reshape(-1, 1))
    spO2_scaler.fit(all_train_spO2_data.reshape(-1, 1))
    artmap_scaler.fit(all_train_artmap_data.reshape(-1, 1))

    return {"BtO2": btO2_scaler, "HR": hr_scaler, "SpO2": spO2_scaler, "artMAP": artmap_scaler}


def identify_extreme_subjects(all_data):
    """
    Identifies the subjects with the most extreme of each feature so the observation with that point can be put in the
    training set, ensuring the normalized CV and test sets will be within the range defined by the most extreme features.
    :param all_data: DataFrame
    :return: Dictionary. Key: Subject number of a subject with the most extreme of a feature. Value: The time of the row
             with the most extreme of a feature for the subject.
    """
    def max_row(all_data, col):
        return all_data.loc[all_data[col] == all_data[col].max()].iloc[0]

    def min_row(all_data, col):
        return all_data.loc[all_data[col] == all_data[col].min()].iloc[0]

    def get_subject_number(row):
        return int(row["subject_id"].lstrip("confocal_"))

    def add_to_dictionary(dictionary, row):
        subject_number = get_subject_number(row)
        if subject_number in dictionary:
            dictionary[subject_number] = max(row["time"], dictionary[subject_number])  # Store the later time
        else:
            dictionary[subject_number] = row["time"]

    min_date_for_subject = {}

    add_to_dictionary(min_date_for_subject, max_row(all_data, "BtO2"))
    add_to_dictionary(min_date_for_subject, max_row(all_data, "HR"))
    add_to_dictionary(min_date_for_subject, max_row(all_data, "SpO2"))
    add_to_dictionary(min_date_for_subject, max_row(all_data, "artMAP"))

    add_to_dictionary(min_date_for_subject, min_row(all_data, "BtO2"))
    add_to_dictionary(min_date_for_subject, min_row(all_data, "HR"))
    add_to_dictionary(min_date_for_subject, min_row(all_data, "SpO2"))
    add_to_dictionary(min_date_for_subject, min_row(all_data, "artMAP"))

    return min_date_for_subject


def normalize(data, min_max_scalers):
    """
    Normalizes data into a 0 to 1 range. Some subjects' data span different ranges than others' so we normalize
    all the data together independent of what subject contributed the data point. This way, the model can learn when a
    subject's data is larger than the others'.

    It is assumed each of the feature columns being normalized has at least one non-Nan value.

    :param data: Dictionary. Key: Subject number. Value: Data frame of the data for the subject.
    :param min_max_scalers: Dictionary. Key: String, feature to scale. Value: MinMaxScaler
    :return:
    """
    for subject_number in data.keys():
        for feature, scaler in min_max_scalers.items():
            # Some entries may only contain NaNs
            if data[subject_number][feature].count() == 0:
                continue

            # Normalize the feature and assign it back to the same column, ignoring NaNs
            null_index = data[subject_number][feature].isnull()
            to_transform = np.array(data[subject_number][feature].loc[~null_index].tolist()).reshape(-1, 1)
            normalized = scaler.transform(to_transform)
            data[subject_number].loc[~null_index, feature] = normalized
    return data
