
from dataclasses import dataclass
from typing import Optional, Dict, Any
import os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np
from .metrics import compute_metrics

@dataclass
class RFConfig:
    n_estimators: int = 300
    max_depth: Optional[int] = None
    n_jobs: int = -1
    random_state: int = 42

def train_rf(X_tr, y_tr, X_val, y_val, cfg: RFConfig):
    clf = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    )
    clf.fit(X_tr, y_tr)
    pred_val = clf.predict(X_val)
    m = compute_metrics(y_val, pred_val)
    return clf, m

def cross_validate_rf(X, y, cfg: RFConfig, cv_splits=3):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cfg.random_state)
    acc = cross_val_score(RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
    ), X, y, cv=skf, scoring="accuracy")
    return {"cv_accuracy_mean": float(acc.mean()), "cv_accuracy_std": float(acc.std())}

def save_rf(model, path):
    joblib.dump(model, path)

def load_rf(path):
    return joblib.load(path)
