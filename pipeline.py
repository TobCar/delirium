"""
@author: Tobias Carryer
"""

import pandas as pd
import numpy as np
import warnings
import os
import pickle
import tensorflow as tf
from simple_model import create_model
from feature_label_generation import generate_compound_image_feature_label_pairs, calculate_steps_per_epoch
from keras.optimizers import Adam
from keras.metrics import binary_accuracy
from keras.losses import binary_crossentropy
from dropping_subjects import drop_some_subjects
from splitting_data import get_data_split_up, test_set
from labels_dictionary import labels
from keras.callbacks import EarlyStopping
from learning_rate_tracker import LearningRateTracker
from normalizing_data import create_min_max_scalers, normalize, identify_extreme_subjects
from sklearn.externals import joblib
from sklearn.utils.class_weight import compute_class_weight

# Seed for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)

# Make TensorFlow use one thread, multiple threads are a source of unpredictability
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
tf.set_random_seed(42)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# Define hyperparameters
compound_image_size = 128
observation_size = compound_image_size
epochs = 100  # Early stopping stops training before the 100 epochs most of the time
number_of_features = 4
batch_size = 32

print("Reading in the data")
all_subject_data = pd.read_csv("confocal_all_patient_phys_data.txt", sep="\t")

print("Dropping some patients")
subject_data = drop_some_subjects(all_subject_data)

print("Normalizing data")
all_train_data = subject_data
for subject_number in test_set:
    all_train_data = all_train_data[all_train_data["subject_id"] != "confocal_%d" % subject_number]

scalers = create_min_max_scalers(all_train_data)
subject_data = normalize(subject_data, scalers)

print("Splitting the data")
min_date_for_subject = identify_extreme_subjects(subject_data)
(train_data, train_lbls), (cv_data, cv_lbls), (cv_data_for_evaluating, cv_lbls_for_evaluating), (test_data, test_lbls) = get_data_split_up(subject_data, labels, min_date_for_subject, observation_size)

print("Saving MinMaxScalers")
# The same MinMaxScalers must be used when the model is used on the test set
extension = ".joblib"
for feature, scaler in scalers.items():
    filename = feature + extension
    joblib.dump(scaler, filename)

print("Calculating class weights")
sklearn_class_weights = compute_class_weight('balanced', [0, 1], train_lbls)
class_weights = {0: sklearn_class_weights[0], 1: sklearn_class_weights[1]}

print("Creating the model")
model = create_model(compound_image_size, number_of_features*3)

print("Compiling the model")
model.compile(optimizer=Adam(), loss=binary_crossentropy, metrics=[binary_accuracy])

print("Training the model")
lrate_tracker = LearningRateTracker()
early_stopping = EarlyStopping("val_loss", patience=5, restore_best_weights=True)
callbacks_list = [lrate_tracker, early_stopping]

training_generator = generate_compound_image_feature_label_pairs(train_data, train_lbls,
                                                                 image_size=compound_image_size,
                                                                 batch_size=batch_size)
validation_generator = generate_compound_image_feature_label_pairs(cv_data, cv_lbls,
                                                                   image_size=compound_image_size,
                                                                   batch_size=batch_size)

steps_per_epoch = calculate_steps_per_epoch(train_data, batch_size=batch_size)
validation_steps_per_epoch = calculate_steps_per_epoch(cv_data, batch_size=batch_size)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    history = model.fit_generator(training_generator, epochs=epochs, steps_per_epoch=steps_per_epoch,
                                  validation_data=validation_generator, validation_steps=validation_steps_per_epoch,
                                  callbacks=callbacks_list, class_weight=class_weights)

    print("Saving the history")
    np.save("learning_rate_history.npy", lrate_tracker.learning_rates)
    curr_working_dir = os.getcwd()
    with open(os.path.join(curr_working_dir, 'training_history_dict.pickle'), 'wb') as f:
        pickle.dump(history.history, f)

print("Saving model weights")
model.save_weights("model.h5")
