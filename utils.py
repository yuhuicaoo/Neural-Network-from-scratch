import numpy as np


def xavier_uniform(shape):
    """ """
    x = np.sqrt(6 / np.sum(shape))
    return np.random.uniform(-x, x, size=shape)


def xavier_normal(shape):
    """ """
    standard_dev = np.sqrt(2 / np.sum(shape))
    return np.random.normal(0, standard_dev, size=shape)
