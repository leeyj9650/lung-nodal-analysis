"""
src/utils/metrics.py

역할
- binary classification 평가 지표 계산
- accuracy, macro_f1, auc
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix


def compute_binary_metrics(y_true, y_prob, y_pred):
    """
    y_true: [N]
    y_prob: [N, 2] softmax probability
    y_pred: [N]
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = float("nan")

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "auc": auc,
    }


def print_binary_report(y_true, y_pred):
    target_names = ["benign_0", "malignant_1"]

    print("\nClassification report:")
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\n")