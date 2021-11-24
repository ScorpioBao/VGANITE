import numpy
import numpy as np


def PEHE(y, y_hat):

    PEHE_val = np.sqrt(np.mean(np.square((y[:, 1] - y[:, 0]) - (y_hat[:, 1] - y_hat[:, 0]))))
    return PEHE_val

def ATE(y,y_hat):
    ATE_val = np.abs(np.mean(y[:, 1] - y[:, 0]) - np.mean(y_hat[:, 1] - y_hat[:, 0]))
    return ATE_val