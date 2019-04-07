"""
@author: Tobias Carryer
"""

import pandas as pd
from dropping_subjects import drop_some_subjects
from splitting_data import get_data_split_up
from sklearn.externals import joblib
from normalizing_data import normalize, identify_extreme_subjects
from labels_dictionary import labels
from simple_model import create_model
from feature_label_generation import generate_compound_image_feature_label_pairs, calculate_steps_per_epoch
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
import warnings


compound_image_size = 128
observation_size = compound_image_size
number_of_features = 4
batch_size = 32

print("Reading in the data")
all_subject_data = pd.read_csv("confocal_all_patient_phys_data.txt", sep="\t")

print("Dropping some patients")
subject_data = drop_some_subjects(all_subject_data)

print("Normalizing data")
scalers = {"SpO2": joblib.load("SpO2.joblib"), "HR": joblib.load("HR.joblib"), "BtO2": joblib.load("BtO2.joblib"),
           "artMAP": joblib.load("artMAP.joblib")}
subject_data = normalize(subject_data, scalers)

print("Splitting the data")
min_date_for_subject = identify_extreme_subjects(subject_data)
_, _, (cv_data_for_evaluating, cv_lbls_for_evaluating), (test_data, test_lbls) = get_data_split_up(subject_data, labels, min_date_for_subject, observation_size)

print("Loading the model")
model = create_model(compound_image_size, number_of_features*3)
model.load_weights("model.h5")

print("Compiling the model")
model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=[binary_accuracy])

print("Testing the model")
data_to_test_on = test_data  # Change to the data set you want to evaluate metrics for
labels_for_data = test_lbls  # Change to the labels corresponding to the data you are testing on

accuracies = []
for subject_number in test_data.keys():
    data = test_data[subject_number]
    lbls = test_lbls[subject_number]

    print("Evaluating for subject %d" % subject_number)
    
    generator = generate_compound_image_feature_label_pairs(data, lbls,
                                                            image_size=compound_image_size,
                                                            batch_size=batch_size)
    steps_per_epoch = calculate_steps_per_epoch(data_to_test_on, batch_size=batch_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = model.evaluate_generator(generator, steps=steps_per_epoch)
        # print(model.metrics_names[0] + " " + str(outputs[0]))  # Uncomment to print the evaluated loss
        print(model.metrics_names[1] + " " + str(outputs[1]))
        accuracies.append(outputs[1])

print("Average accuracy (per subject): " + str(sum(accuracies) / len(accuracies)))
