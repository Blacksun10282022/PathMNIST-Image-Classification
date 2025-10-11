
import os
import numpy as np
from typing import Union
from tensorflow import keras

def load_model(model_path: str):
    return keras.models.load_model(model_path)

def predict_numpy(model, X: np.ndarray):
    if X.dtype != np.float32 and X.max() > 1.0:
        X = X.astype("float32") / 255.0
    if X.ndim == 3:
        X = X[None, ...]
    probs = model.predict(X, verbose=0)
    return probs.argmax(axis=1), probs
