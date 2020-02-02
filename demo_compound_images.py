import numpy as np
import pandas as pd
from pyts.image import GASF, GADF, MTF
from splitting_data import get_subject_data
from matplotlib import pyplot as plt


def create_gasf_gadf_mtf_compound_images(observations, image_size=128):
    """
    Designed to take observations of time series data and create compound images from it to analyze with a CNN.
    The research paper that came up with GASF-GADF-MTF images can be read here: https://arxiv.org/pdf/1506.00327.pdf

    :param observations: A 2D array. Shape: [n_observations, observation_window_length]
    :param image_size: Size of the images to create. Must be equal to or smaller than the length of the
                       time series data in each observation.
    :raises ValueError: If observations is empty.
    :return: An array of images ready to be used in a CNN. Shape: [n_observations, image_size, image_size, 3]
             The origin of each image is the top-left corner. When plotted, it would be the point (0,0).
    """
    if len(observations) == 0:
        raise ValueError("Observations cannot be empty.")

    gasf_transformer = GASF(image_size)
    gadf_transformer = GADF(image_size)
    mtf_transformer = MTF(image_size)

    gasf = gasf_transformer.fit_transform(observations)
    gadf = gadf_transformer.fit_transform(observations)
    mtf = mtf_transformer.fit_transform(observations)

    return np.stack((gasf, gadf, mtf), axis=3)


if __name__ == "__main__":
    subject_num_to_study = 4
    feature_to_study = "HR"

    # Generate a compound image and display it to the user
    all_subject_data = pd.read_csv("confocal_all_patient_phys_data.txt", sep="\t")
    subject_data = get_subject_data(all_subject_data, subject_num_to_study)

    i = 0
    observation = subject_data[feature_to_study].iloc[i:i+128]
    while observation.isnull().values.any():
        observation = subject_data[feature_to_study].iloc[i:i+128]
        i += 1
    observation = observation.values
    observations = [observation]

    images = create_gasf_gadf_mtf_compound_images(observations, image_size=128)

    gasf = images[:,:,:,0]
    gadf = images[:,:,:,1]
    mtf = images[:,:,:,2]

    plt.figure(figsize=(8, 8))

    plt.subplot(221)
    plt.imshow(gasf[0], cmap='rainbow')
    plt.title("Gramian Angular Summation Field", fontsize=8)
    plt.tick_params(axis='x', colors=(0, 0, 0, 0))
    plt.tick_params(axis='y', colors=(0, 0, 0, 0))

    plt.subplot(222)
    plt.imshow(gadf[0], cmap='rainbow')
    plt.title("Gramian Angular Difference Field", fontsize=8)
    plt.tick_params(axis='x', colors=(0, 0, 0, 0))
    plt.tick_params(axis='y', colors=(0, 0, 0, 0))

    plt.subplot(223)
    plt.imshow(mtf[0], cmap='rainbow')
    plt.title("Markov Transition Field", fontsize=8)
    plt.tick_params(axis='x', colors=(0, 0, 0, 0))
    plt.tick_params(axis='y', colors=(0, 0, 0, 0))

    plt.subplot(224)
    plt.plot(observation)
    plt.title("Heart Rate", fontsize=8)

    plt.suptitle("Fields generated for a window of heart rate data")

    plt.show()
