"""
@author: Tobias Carryer, Shreyansh Anand
"""

import pandas as pd
from confusion_matrix import confusion_matrix_creator
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

predicted_vals = []
final_labels = []
accuracies = []
for subject_number in data_to_test_on.keys():
    data = data_to_test_on[subject_number]
    lbls = labels_for_data[subject_number]

    print("Evaluating for subject %d" % subject_number)

    # Evaluating and predicting are thread safe, requiring their own identical generators
    generator_1 = generate_compound_image_feature_label_pairs(data, lbls,
                                                              image_size=compound_image_size,
                                                              batch_size=batch_size)
    generator_2 = generate_compound_image_feature_label_pairs(data, lbls,
                                                              image_size=compound_image_size,
                                                              batch_size=batch_size)
    steps_per_epoch = calculate_steps_per_epoch(data_to_test_on, batch_size=batch_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Accuracy
        outputs = model.evaluate_generator(generator_1, steps=steps_per_epoch)
        print(model.metrics_names[1] + " " + str(outputs[1]))
        accuracies.append(outputs[1])

        # Sensitivity and Specificity
        predictions = model.predict_generator(generator_2, steps=steps_per_epoch)
        rounded_predictions = int(round((sum(predictions)[0]) / len(predictions)))
        predicted_vals.append(rounded_predictions)
        final_labels.append(lbls[0])

print("Average accuracy (per subject): " + str(sum(accuracies) / len(accuracies)))
confusion_matrix_creator(final_labels, predicted_vals)
