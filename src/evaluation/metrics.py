# test dataset 평가  : accuracy / AUC / F1 / sensitivity / specificity


import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix
)

# =========================================================
# 모든 metric 계산 전담
# =========================================================
def compute_metrics(labels, probs, preds):

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "auc": float(auc),
        "accuracy": float(acc),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)
    }, cm