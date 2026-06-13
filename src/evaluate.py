import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    score = np.sum(
        np.where(diff<0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    )
    return score