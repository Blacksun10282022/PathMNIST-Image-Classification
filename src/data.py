
import os
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def _load_pair(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(data_dir, "y_train.npy"), allow_pickle=True)
    X_test  = np.load(os.path.join(data_dir, "X_test.npy"), allow_pickle=True)
    y_test  = np.load(os.path.join(data_dir, "y_test.npy"), allow_pickle=True)
    return X_train, y_train, X_test, y_test

def load_data(data_dir="data", use_sample=False, val_size=0.1, for_cnn=True, seed=42):
    ddir = os.path.join(data_dir, "samples") if use_sample else data_dir
    X_train, y_train, X_test, y_test = _load_pair(ddir)

    # Train/Val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    idx_train, idx_val = next(sss.split(X_train, y_train))
    X_tr, y_tr = X_train[idx_train], y_train[idx_train]
    X_val, y_val = X_train[idx_val], y_train[idx_val]

    if for_cnn:
        # normalize to [0,1]
        X_tr = X_tr.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0
    else:
        # flatten to [N, D]
        X_tr = X_tr.reshape((X_tr.shape[0], -1))
        X_val = X_val.reshape((X_val.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

    return (X_tr, y_tr), (X_val, y_val), (X_test, y_test)
