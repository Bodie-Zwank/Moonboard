import numpy as np

def mse(expected_output, actual_output):
    return np.mean(np.power(expected_output - actual_output, 2))

def mse_prime(expected_output, actual_output):
    # chain rule switches order of actual output and expected output in derivative (because the sign flips)
    return 2 * (actual_output - expected_output) / np.size(expected_output)

# binary cross entropy
def bce(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def bce_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)