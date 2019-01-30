"""
@author: Tobias Carryer
"""

import pandas as pd
import numpy as np
import warnings
import os
import pickle
from densenet_14 import create_densenet
from feature_label_generation import generate_compound_image_feature_label_pairs, calculate_steps_per_epoch
from keras.optimizers import SGD
from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy
from dropping_subjects import drop_some_subjects
from filling_in_data import fill_in_data
from splitting_data import get_data_split_up
from labels_dictionary import labels
from keras.callbacks import LearningRateScheduler
from step_decay import step_decay
from learning_rate_tracker import LearningRateTracker
from weighing_subjects import weigh_subjects


# Define hyperparameters
compound_image_size = 128
observation_size = compound_image_size
epochs = 50
number_of_features = 4
batch_size = 32
initial_learning_rate = 0.1

print("Reading in the data")
all_subject_data = pd.read_csv("confocal_all_patient_phys_data.txt", sep="\t")

print("Dropping some patients")
all_subject_data = drop_some_subjects(all_subject_data)

print("Splitting the data")
np.random.seed(131313)  # Set a seed so random splits are the same when this script is run multiple times
train_data, cv_data, test_data = get_data_split_up(all_subject_data, labels)

print("Calculating weights for each patient")
subject_weights = weigh_subjects(train_data)

print("Filling in missing values")
train_data = fill_in_data(train_data)
cv_data = fill_in_data(cv_data)
test_data = fill_in_data(test_data)

print("Creating the model")
model = create_densenet(image_size=compound_image_size, number_of_channels=number_of_features*3)

print("Compiling the model")
model.compile(optimizer=SGD(lr=initial_learning_rate), loss=binary_crossentropy, metrics=[binary_accuracy])

print("Training the model")
lrate_scheduler = LearningRateScheduler(step_decay)  # Learning rate is updated by the learning rate scheduler
lrate_tracker = LearningRateTracker()
callbacks_list = [lrate_scheduler, lrate_tracker]

training_generator = generate_compound_image_feature_label_pairs(train_data, labels, subject_weights,
                                                                 observation_size=observation_size,
                                                                 image_size=compound_image_size,
                                                                 batch_size=batch_size)
validation_generator = generate_compound_image_feature_label_pairs(cv_data, labels,
                                                                   observation_size=observation_size,
                                                                   image_size=compound_image_size,
                                                                   batch_size=batch_size)

steps_per_epoch = calculate_steps_per_epoch(train_data, observation_size=observation_size, batch_size=batch_size)
validation_steps_per_epoch = calculate_steps_per_epoch(cv_data, observation_size=observation_size, batch_size=batch_size)

with warnings.catch_warnings():
    # Ignore a FutureWarning from the image generation
    # pyts/image/image.py:321: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated;
    # use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index,
    # `arr[np.array(seq)]`, which will result either in an error or a different result.
    # MTF[np.meshgrid(list_values[i], list_values[j])] = MTM[i, j]
    warnings.simplefilter("ignore")
    history = model.fit_generator(training_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                                  callbacks=callbacks_list)

    print("Saving the history")
    np.save("learning_rate_history.npy", lrate_tracker.learning_rates)
    curr_working_dir = os.getcwd()
    with open(os.path.join(curr_working_dir, 'training_history_dict.pickle'), 'wb') as f:
        pickle.dump(history.history, f)

print("Saving the model")
model.save_weights("model.h5")
