"""
@author: Tobias Carryer
"""


def step_decay(epoch, lr):
    """
    Initial learning rate is 0.1. The learning rate is reduced by a factor of 0.1 every 10 epochs. For example,
    the ninth epoch's learning rate would be 0.1 and the tenth epoch's learning rate would be 0.05.

    This function can be used with LearningRateScheduler as a callback like so:
    ``lr = LearningRateScheduler(step_decay)``
    Step decay should be called before the epoch begins. LearningRateScheduler takes care of this.

    :param epoch: The current epoch. (The first epoch is zero.)
    :param lr: The learning rate.
    :return: The learning rate, updated if necessary.
    """
    drop = 0.5
    epochs_drop = 10.0
    if (epoch+1) % epochs_drop == 0 and epoch != 0:
        lr *= drop
    return lr
