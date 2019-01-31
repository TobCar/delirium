"""
@author: Tobias Carryer
"""


def weigh_subjects(to_weigh, labels):
    """
    Some subjects have more data than others. This function calculates weights we can use to have our model
    weigh them equally while training so the subjects with more data do not bias the results. The process is repeated
    for each label separately.
    :param to_weigh: Dictionary. Key: Subject number. Value: Data frame of the subject's data.
    :param labels: Dictionary. Key: Subject number. Value: Label for the subject.
    :return: Dictionary. Key: Subject number. Value: Weight for each training case made up of data from the subject.
    """
    n_delirious_data_points = 0
    n_delirious_data_points_for_subject = {}
    n_non_delirious_data_points = 0
    n_non_delirious_data_points_for_subject = {}

    for subject_number, subject_data in to_weigh.items():
        if labels[subject_number] == 1:
            n_delirious_data_points += subject_data.shape[0]
            n_delirious_data_points_for_subject[subject_number] = subject_data.shape[0]
        else:
            n_non_delirious_data_points += subject_data.shape[0]
            n_non_delirious_data_points_for_subject[subject_number] = subject_data.shape[0]

    weights = {}

    for subject_number, subject_data_points in n_delirious_data_points_for_subject.items():
        # Inverse of the percentage of data made up by a subject
        weights[subject_number] = n_delirious_data_points / subject_data_points

    for subject_number, subject_data_points in n_non_delirious_data_points_for_subject.items():
        # Inverse of the percentage of data made up by a subject
        weights[subject_number] = n_non_delirious_data_points / subject_data_points

    return weights
