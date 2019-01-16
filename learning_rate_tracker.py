from keras.callbacks import Callback
import keras.backend as K


class LearningRateTracker(Callback):

    def __init__(self):
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr)  # Get a value from the tensor object
        self.learning_rates.append(lr)
