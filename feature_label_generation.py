"""
@author: Tobias Carryer
"""

from create_array_with_rolling_window import create_array_with_rolling_window
from pyts.image import GASF, GADF, MTF
import numpy as np
import math


def generate_compound_image_feature_label_pairs(data, labels, observation_size=128, image_size=128, batch_size=1024):
    """
    Generate GASF-GADF-MTF compound images and pair them with the label for the image's observation period.
    This function is intended to be used with fit_generator() in Keras.

    :param data: Dictionary. Key: Integer. Value: Data frame of subject data.
    :param labels: Dictionary. Key: Integer. Value: Integer.
    :param observation_size: Size of the look back period
    :param image_size: Size of the images to generate. Must be equal to or smaller than the length of the
                       time series data in each observation.
    :param batch_size: The number of images generated per yield
    :return: Yields a tuple (compound_images, labels) for data sets with enough data points for at least one
             observation and with labels associated with the data.
    """
    for subject_number, subject_data in data.items():
        # Shape: [n_observations, observation_window_length]
        btO2_observations = create_array_with_rolling_window(subject_data["BtO2"], window_size=observation_size)
        hr_observations = create_array_with_rolling_window(subject_data["HR"], window_size=observation_size)
        spO2_observations = create_array_with_rolling_window(subject_data["SpO2"], window_size=observation_size)
        artmap_observations = create_array_with_rolling_window(subject_data["artMAP"], window_size=observation_size)

        btO2_observations,\
        hr_observations,\
        spO2_observations,\
        artmap_observations = remove_cases_with_nan(btO2_observations, hr_observations, spO2_observations,
                                                    artmap_observations)

        # Skip subjects with no observations after removing observations with NaNs
        if len(btO2_observations) == 0 or len(hr_observations) == 0 or len(spO2_observations) == 0\
                or len(artmap_observations) == 0:
            continue

        btO2_image_generator = generate_gasf_gadf_mtf_compound_images(btO2_observations, image_size=image_size,
                                                                      batch_size=batch_size)
        hr_image_generator = generate_gasf_gadf_mtf_compound_images(hr_observations, image_size=image_size,
                                                                    batch_size=batch_size)
        spO2_image_generator = generate_gasf_gadf_mtf_compound_images(spO2_observations, image_size=image_size,
                                                                      batch_size=batch_size)
        artmap_image_generator = generate_gasf_gadf_mtf_compound_images(artmap_observations, image_size=image_size,
                                                                        batch_size=batch_size)

        image_generators = zip(btO2_image_generator, hr_image_generator, spO2_image_generator, artmap_image_generator)

        for btO2_image, hr_image, spO2_image, artmap_image, in image_generators:
            compound_images = np.concatenate((btO2_image, hr_image, spO2_image, artmap_image), axis=3)
            labels_to_yield = np.repeat(labels[subject_number], compound_images.shape[0])
            yield (compound_images, labels_to_yield)


def generate_gasf_gadf_mtf_compound_images(observations, image_size=128, batch_size=64):
    """
    Designed to take observations of time series data and generate compound images from it to analyze with a CNN.
    The research paper that came up with GASF-GADF-MTF images can be read here: https://arxiv.org/pdf/1506.00327.pdf

    :param observations: A read-only 2D numpy array. Shape: [n_observations, observation_window_length]
    :param image_size: Size of the images to generate. Must be equal to or smaller than the length of the
                       time series data in each observation.
    :param batch_size: The number of images generated per yield
    :raises ValueError: If observations is empty.
    :return: Yields an array of images ready to be used in a CNN. Shape: [batch_size, image_size, image_size, 3]
             If there are fewer observations left to generate images for, the batch size may be less than expected.
             The origin of each image is the top-left corner. When plotted, it would be the point (0,0).
    """
    if len(observations) == 0:
        raise ValueError("Observations cannot be empty.")

    gasf_transformer = GASF(image_size)
    gadf_transformer = GADF(image_size)
    mtf_transformer = MTF(image_size)

    # Split up the image generation into smaller batches to handle
    upper_bound = min(len(observations), batch_size)
    lower_bound = 0
    while lower_bound < len(observations):
        observations_batch = observations[lower_bound:upper_bound]

        gasf = gasf_transformer.fit_transform(observations_batch)
        gadf = gadf_transformer.fit_transform(observations_batch)
        mtf = mtf_transformer.fit_transform(observations_batch)

        yield np.stack((gasf, gadf, mtf), axis=3)

        lower_bound = upper_bound
        upper_bound += batch_size
        upper_bound = min(len(observations), upper_bound)


def remove_cases_with_nan(btO2_observations, hr_observations, spO2_observations, artmap_observations):
    """
    :param btO2_observations:
    :param hr_observations:
    :param spO2_observations:
    :param artmap_observations:
    :return: The observations passed in as parameters, without observations where at least one of the values for one of
             the features is a NaN. It is assumed the features are in sync (ex. when we use btO2_observations[i] we
             use hr_observations[i] in that same training case).
    """
    # Logical arrays of what observations do not contain NaNs
    btO2_observations_not_nan = ~np.isnan(btO2_observations).any(axis=1)  # axis=1 is the contents of an observation
    hr_observations_not_nan = ~np.isnan(hr_observations).any(axis=1)
    spO2_observations_not_nan = ~np.isnan(spO2_observations).any(axis=1)
    artmap_observations_not_nan = ~np.isnan(artmap_observations).any(axis=1)

    # Logical and between all the logical arrays to get what observations do not have any NaNs in any feature
    observations_with_no_nans = np.logical_and(np.logical_and(np.logical_and(btO2_observations_not_nan,
                                                                             hr_observations_not_nan),
                                                              spO2_observations_not_nan),
                                               artmap_observations_not_nan)

    return btO2_observations[observations_with_no_nans], hr_observations[observations_with_no_nans],\
           spO2_observations[observations_with_no_nans], artmap_observations[observations_with_no_nans]


def calculate_steps_per_epoch(data, observation_size=128, batch_size=64):
    """
    :param data: Dictionary. Key: Integer. Value: Data frame of subject data.
    :param observation_size: Size of the look back period
    :param batch_size: The number of images generated per yield
    :return: How many times the generator will yield
    """
    yields = 0
    for subject_number, subject_data in data.items():
        btO2_observations = create_array_with_rolling_window(subject_data["BtO2"], window_size=observation_size)
        hr_observations = create_array_with_rolling_window(subject_data["HR"], window_size=observation_size)
        spO2_observations = create_array_with_rolling_window(subject_data["SpO2"], window_size=observation_size)
        artmap_observations = create_array_with_rolling_window(subject_data["artMAP"], window_size=observation_size)

        valid_observations = 0
        for i in range(len(btO2_observations)):
            btO2_has_nan = np.isnan(btO2_observations[i]).any()
            hr_has_nan = np.isnan(hr_observations[i]).any()
            spO2_has_nan = np.isnan(spO2_observations[i]).any()
            artmap_has_nan = np.isnan(artmap_observations[i]).any()

            # A feature-label pair is generated if there were no NaNs in any of the feature observations
            if not btO2_has_nan and not hr_has_nan and not spO2_has_nan and not artmap_has_nan:
                valid_observations += 1

        yields += math.ceil(valid_observations / batch_size)
    return yields
