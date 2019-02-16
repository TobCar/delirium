"""
@author: Tobias Carryer
"""

import numpy as np
import pandas as pd
from dropping_subjects import drop_some_subjects
from splitting_data import get_data_split_up
from sklearn.externals import joblib
from normalizing_data import normalize, identify_extreme_subjects
from labels_dictionary import labels
from sklearn.utils.class_weight import compute_class_weight
from weighing_subjects import weigh_subjects
from testing_the_model import test_model
from simple_model import create_model


compound_image_size = 128
observation_size = compound_image_size
number_of_features = 4

print("Loading the model")
model = create_model(compound_image_size, number_of_features*3)
model.load_weights("model.h5")

print("Reading in the data")
all_subject_data = pd.read_csv("confocal_all_patient_phys_data.txt", sep="\t")

print("Dropping some patients")
subject_data = drop_some_subjects(all_subject_data)

print("Splitting the data")
min_date_for_subject = identify_extreme_subjects(subject_data)
train_data, cv_data, test_data = get_data_split_up(subject_data, labels, min_date_for_subject, observation_size)

print("Normalizing data")
scalers = {"SpO2": joblib.load("SpO2.joblib"), "HR": joblib.load("HR.joblib"), "BtO2": joblib.load("BtO2.joblib"),
           "artMAP": joblib.load("artMAP.joblib")}
train_data = normalize(train_data, scalers)
cv_data = normalize(cv_data, scalers)
test_data = normalize(test_data, scalers)

print("Calculating class and subject weights")
labels_to_balance = [labels[subject_number] for subject_number in train_data.keys()]
sklearn_class_weights = compute_class_weight('balanced', [0, 1], labels_to_balance)
class_weights = {0: sklearn_class_weights[0], 1: sklearn_class_weights[1]}
subject_weights = weigh_subjects(train_data, labels)

print("Testing the model")
accuracy, voting_accuracy = test_model(model, cv_data, compound_image_size=compound_image_size,
                                       observation_size=observation_size)

def average(dictionary):
    sum = 0
    for val in dictionary.values():
        sum += val
    return sum / len(dictionary)
print("Average Accuracy: "+str(average(accuracy)))
print("Average Accuracy with Voting: "+str(average(accuracy)))
