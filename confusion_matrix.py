"""
@author: Shreyansh Anand
"""

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools


def confusion_matrix_creator(test_labels, predicted_values):
    """
    :param predicted_values: the predicted value for whether a patient has delirium or not (rounded, not actual probability)
    :param test_labels: the test labels for each of the patients
    :return:
    """
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_values).ravel()
    confusion_matrix_to_plot = np.array([[tp, fp], [fn, tn]])
    cm_plot_labels = ['Delirium', 'No Delirium']
    plot_confusion_matrix(confusion_matrix_to_plot, cm_plot_labels)
    plt.show()


def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Predicted label')
    plt.xlabel("True label")
