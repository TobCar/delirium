"""
@author: Tobias Carryer
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_min_max_scalers(subject_data):
    """
    Creates the MinMaxScalers necessary to normalize the data.
    :param subject_data: DataFrame
    :return: Dictionary. Key: String, feature to normalize. Value: MinMaxScaler that can normalize the feature.
    """
    # We could create one scaler for all the features but this approach makes it easier to use the scalers outside
    # of this project if someone needs them.
    all_btO2_data = np.array(subject_data["BtO2"].tolist())
    all_hr_data = np.array(subject_data["HR"].tolist())
    all_spO2_data = np.array(subject_data["SpO2"].tolist())
    all_artmap_data = np.array(subject_data["artMAP"].tolist())

    btO2_scaler = MinMaxScaler()
    hr_scaler = MinMaxScaler()
    spO2_scaler = MinMaxScaler()
    artmap_scaler = MinMaxScaler()

    btO2_scaler.fit(all_btO2_data.reshape(-1, 1))
    hr_scaler.fit(all_hr_data.reshape(-1, 1))
    spO2_scaler.fit(all_spO2_data.reshape(-1, 1))
    artmap_scaler.fit(all_artmap_data.reshape(-1, 1))

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

    :param data: DataFrame
    :param min_max_scalers: Dictionary. Key: String, feature to scale. Value: MinMaxScaler
    :return: Normalized data
    """
    for feature, scaler in min_max_scalers.items():
        # Normalize the feature and assign it back to the same column, ignoring NaNs
        null_index = data[feature].isnull()
        to_transform = np.array(data[feature].loc[~null_index].tolist()).reshape(-1, 1)
        normalized = scaler.transform(to_transform)
        data.loc[~null_index, feature] = normalized
    return data
