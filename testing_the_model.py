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
from simple_model import create_model
from feature_label_generation import generate_compound_image_feature_label_pairs, calculate_steps_per_epoch
from keras.optimizers import SGD
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
all_subject_data = drop_some_subjects(all_subject_data)

print("Splitting the data")
np.random.seed(7777777)  # Set a seed so random splits are the same when this script is run multiple times
must_go_in_training = identify_extreme_subjects(all_subject_data)
train_data, cv_data, test_data = get_data_split_up(all_subject_data, labels, must_go_in_training)

print("Normalizing data")
scalers = {"SpO2": joblib.load("SpO2.joblib"), "HR": joblib.load("HR.joblib"), "BtO2": joblib.load("BtO2.joblib"),
           "artMAP": joblib.load("artMAP.joblib")}
train_data = normalize(train_data, scalers)
cv_data = normalize(cv_data, scalers)
test_data = normalize(test_data, scalers)

print("Loading the model")
model = create_model(compound_image_size, number_of_features*3)
model.load_weights("model.h5")

print("Compiling the model")
model.compile(optimizer=SGD(), loss=binary_crossentropy, metrics=[binary_accuracy])

print("Testing the model")
data_to_test_on = train_data  # Change to the data set you want to evaluate metrics for

generator = generate_compound_image_feature_label_pairs(data_to_test_on, labels,
                                                        observation_size=observation_size,
                                                        image_size=compound_image_size,
                                                        batch_size=batch_size)
steps_per_epoch = calculate_steps_per_epoch(data_to_test_on, observation_size=observation_size, batch_size=batch_size)

with warnings.catch_warnings():
    # Ignore a FutureWarning from the image generation
    # pyts/image/image.py:321: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated;
    # use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index,
    # `arr[np.array(seq)]`, which will result either in an error or a different result.
    # MTF[np.meshgrid(list_values[i], list_values[j])] = MTM[i, j]
    warnings.simplefilter("ignore")
    outputs = model.evaluate_generator(generator, steps=steps_per_epoch)
    print(model.metrics_names[0] + " " + str(outputs[0]))
    print(model.metrics_names[1] + " " + str(outputs[1]))
