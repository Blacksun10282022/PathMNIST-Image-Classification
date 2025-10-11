
import os, io, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return {"accuracy": float(acc), "macro_f1": float(f1)}

def save_classification_report(y_true, y_pred, out_path):
    report = classification_report(y_true, y_pred, digits=4)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report

def save_confusion_matrix(y_true, y_pred, out_path, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels if labels is not None else None)
    plt.figure()
    disp.plot(include_values=True, xticks_rotation="vertical", cmap=None)  # default color map
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_training_curve(history, out_path):
    plt.figure()
    plt.plot(history.history.get("loss", []), label="loss")
    if "val_loss" in history.history: plt.plot(history.history["val_loss"], label="val_loss")
    if "accuracy" in history.history: plt.plot(history.history["accuracy"], label="acc")
    if "val_accuracy" in history.history: plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
