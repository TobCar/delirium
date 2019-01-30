"""
@author: Tobias Carryer
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_min_max_scalers(train_data):
    """
    Creates the MinMaxScalers necessary to normalize the data.
    :param train_data: Dictionary. Key: Subject number. Value: Data frame of the data from the subject.
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
            # Normalize the feature and assign it back to the same column, ignoring NaNs
            null_index = data[subject_number][feature].isnull()
            to_transform = np.array(data[subject_number][feature].loc[~null_index].tolist()).reshape(-1, 1)
            normalized = scaler.transform(to_transform)
            data[subject_number].loc[~null_index, feature] = normalized
    return data