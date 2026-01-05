from imports_and_env_setup import *

def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return float(
        np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    )
