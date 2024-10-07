import numpy as np


def categoricalCrossEntropy(y_pred, y_target):
    samples = len(y_pred)

    # clip to prevent log(0) = inf
    y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)

    if len(y_target.shape) == 1:
        # handle scalar class labels
        correct_confidences = y_pred_clipped[range(samples), y_target]
    elif len(y_target.shape) == 2:
        # handle One-hot encoded class labels
        correct_confidences = np.sum(y_pred_clipped * y_target, axis=1)

    loss = -np.log(correct_confidences)
    return loss


def regressionMSE(y_pred, y_target, deriv=False):
    print(np.size(y_pred), y_pred.shape)
    if deriv:
        return 2 * (y_pred - y_target) / np.size(y_pred)
    return np.mean(y_pred - y_target)**2
