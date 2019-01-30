"""
@author: Tobias Carryer
"""


def weigh_subjects(to_weigh):
    """
    Some subjects have more data than others. This function calculates weights we can use to have our model
    weigh them equally while training so the subjects with more data do not bias the results.
    :param to_weigh: Key: Subject number. Value: Data frame of the subject's data.
    :return: Dictionary. Key: Subject number. Value: Weight for each training case made up of data from the subject.
    """
    n_data_points = 0
    n_data_points_for_subject = {}

    for subject_number, subject_data in to_weigh.items():
        n_data_points += subject_data.shape[0]
        n_data_points_for_subject[subject_number] = subject_data.shape[0]

    weights = {}

    for subject_number, subject_data_points in n_data_points_for_subject.items():
        # Inverse of the percentage of data made up by a subject
        weights[subject_number] = n_data_points / subject_data_points

    return weights
