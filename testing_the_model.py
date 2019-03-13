"""
@author: Tobias Carryer
"""

from feature_label_generation import generate_compound_image_feature_label_pairs, calculate_steps_per_epoch
import numpy as np
from voting import vote
import warnings
from labels_dictionary import labels


def test_model(model, data_to_test_on, batch_size=100, compound_image_size=128, observation_size=128):
    """
    :param model: Model to test.
    :param data_to_test_on: Dictionary. Key: subject number. Value: Data frame.
    :param batch_size: Batch size used during classification. Voting is done per batch so the larger the batch,
                       the more noise can be filtered out.
    :param compound_image_size: Must equal the value used during training (pipeline.py)
    :param observation_size: Must equal the value used during training (pipeline.py)
    :return: Dictionaries. Key: Subject number. Value: Float. The first dictionary is the accuracy per subject.
             The second dictionary is the accuracy per subject when voting is used.
    """

    accuracy = {}
    accuracy_with_voting = {}

    for subject_number in data_to_test_on.keys():
        data_to_classify = data_to_test_on[subject_number]
        fake_dict_of_all_patient_data = {subject_number: data_to_classify}

        generator = generate_compound_image_feature_label_pairs(fake_dict_of_all_patient_data, labels,
                                                                           observation_size=observation_size,
                                                                           image_size=compound_image_size,
                                                                           batch_size=batch_size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batches = 0
            total_batches = calculate_steps_per_epoch(fake_dict_of_all_patient_data, batch_size=batch_size)
            observations = 0
            acc = 0
            acc_with_voting = 0
            for images, batch_labels in generator:
                batches += 1
                if batches > total_batches:
                    break  # Seen all batches, exit the generator's endless loop

                predictions = model.predict(images, batch_size)
                predicted_labels = predictions >= 0.5
                acc += count_correct(batch_labels, predicted_labels)

                majority_prediction = vote(predicted_labels)
                acc_with_voting += count_correct(batch_labels, np.repeat(majority_prediction, len(batch_labels)))

                # Used to calculate the percentage accuracy at the end
                observations += len(batch_labels)
            accuracy[subject_number] = acc / observations
            accuracy_with_voting[subject_number] = acc_with_voting / observations

    return accuracy, accuracy_with_voting


def count_correct(labels, predictions):
    correct = 0
    for i in range(len(labels)):
        if labels[i] == predictions[i]:
            correct += 1
    return correct
