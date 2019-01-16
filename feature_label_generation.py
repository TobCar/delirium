"""
@author: Tobias Carryer
"""

from create_array_with_rolling_window import create_array_with_rolling_window
from pyts.image import GASF, GADF, MTF
from multiprocessing.pool import Pool
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

        # Generate the images for the batch and store them to return later
        with Pool(processes=3) as pool:
            gasf_async_result = pool.apply_async(gasf_transformer.fit_transform, (observations_batch,))
            gadf_async_result = pool.apply_async(gadf_transformer.fit_transform, (observations_batch,))
            mtf_async_result = pool.apply_async(mtf_transformer.fit_transform, (observations_batch,))

            gasf_async_result.wait()
            gadf_async_result.wait()
            mtf_async_result.wait()

            yield np.stack((gasf_async_result.get(), gadf_async_result.get(), mtf_async_result.get()), axis=3)

        lower_bound = upper_bound
        upper_bound += batch_size
        upper_bound = min(len(observations), upper_bound)


def calculate_steps_per_epoch(data, observation_size=128, batch_size=64):
    """
    :param data: Dictionary. Key: Integer. Value: Data frame of subject data.
    :param observation_size: Size of the look back period
    :param batch_size: The number of images generated per yield
    :return: How many times the generator will yield
    """
    yields = 0
    for subject_number, subject_data in data.items():
        yields += math.ceil((subject_data.shape[0] - observation_size + 1) / batch_size)
    return yields
