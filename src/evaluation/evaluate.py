# 모델 성능 분석
# 1. 모델 로드          : best_model_a.pth
# 2. ROC curve 생성     : figures/roc_curve.png
# 3. confusion matrix 생서 : figures/confusion_matrix.png


import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from src.configs.config import *
from src.datasets.dataset import get_dataloaders
from src.models.resnet import create_model
from src.evaluation.metrics import compute_metrics


# =========================================================
# device 설정
# =========================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 모델 평가 (pure inference)
# =========================================================
def evaluate_model(model, loader, device):

    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_probs), np.array(all_preds)


# =========================================================
# ROC curve 저장
# =========================================================
def save_roc(labels, probs, method):

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()

    save_path = FIGURE_DIR / f"roc_{method}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] ROC -> {save_path}")


# =========================================================
# Confusion Matrix 저장
# =========================================================
def save_cm(cm, method):

    plt.figure()
    plt.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.xticks([0, 1], ["benign", "malignant"])
    plt.yticks([0, 1], ["benign", "malignant"])

    plt.title("Confusion Matrix")

    save_path = FIGURE_DIR / f"cm_{method}.png"
    plt.savefig(save_path)
    plt.close()

    print(f"[Saved] CM -> {save_path}")


# =========================================================
# main
# =========================================================
def main(method="a"):

    device = get_device()

    print("\n===== EVALUATION =====")
    print("Device:", device)
    print("Method:", method)

    # -------------------------
    # data
    # -------------------------
    _, _, test_loader = get_dataloaders(method)

    # -------------------------
    # model
    # -------------------------
    model = create_model(pretrained=False)
    model = model.to(device)

    model_path = CHECKPOINT_DIR / f"best_model_{method}.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))

    # -------------------------
    # inference
    # -------------------------
    labels, probs, preds = evaluate_model(model, test_loader, device)

    metrics, cm = compute_metrics(labels, probs, preds)

    # -------------------------
    # print result
    # -------------------------
    print("\n===== RESULT =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # -------------------------
    # save json
    # -------------------------
    with open(LOG_DIR / f"eval_{method}.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # -------------------------
    # plots
    # -------------------------
    save_roc(labels, probs, method)
    save_cm(cm, method)

    print("\nEvaluation Done")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="a")
    args = parser.parse_args()

    main(args.method)