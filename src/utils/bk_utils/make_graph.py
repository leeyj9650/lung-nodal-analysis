"""
src/utils/make_graph.py

역할
- 모델의 예측 결과(정답 라벨, 악성 확률)를 바탕으로 ROC Curve 그래프를 그립니다.
- 그래프 이미지(.png)와 수치 데이터(.csv)를 지정된 통합 실험 폴더에 안전하게 저장합니다.
"""

import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def save_roc_curve_and_csv(
    all_true: list,
    all_prob: list,
    final_exp_dir: str,
    time_prefix: str,
    config_part: str,
    split_name: str = "test"
) -> dict:
    """
    정답(all_true)과 예측 확률(all_prob)을 받아 ROC 커브 그래프(.png)와 데이터(.csv)를 저장합니다.
    
    Args:
        all_true: 실제 정답 라벨 목록 (예: [0, 1, 0, 0, 1])
        all_prob: 모델이 예측한 클래스 1(악성)일 확률 목록 (예: [0.12, 0.89, 0.05...])
        final_exp_dir: 결과물이 저장될 최종 통합 폴더 경로
        time_prefix: 파일명 앞에 붙을 타임스탬프 (예: "2606041150")
        config_part: 파일명 뒤에 붙을 실험 조건 세팅 (예: "_fixed_crop_size64")
        split_name: 데이터 영역 구분명 ("train", "val", "test")
        
    Returns:
        saved_paths: 저장된 png와 csv 파일의 경로를 담은 딕셔너리
    """
    # 1. 클래스 1(악성)에 해당하는 확률만 추출 (all_prob가 [prob_0, prob_1] 쌍으로 들어오는 경우 대비)
    if isinstance(all_prob[0], (list, tuple)) and len(all_prob[0]) == 2:
        positive_probs = [p[1] for p in all_prob]
    else:
        positive_probs = all_prob

    # 2. Sklearn 도구를 이용해 민감도(TPR)와 위양성률(FPR) 및 AUC 면적 계산
    fpr, tpr, thresholds = roc_curve(all_true, positive_probs)
    roc_auc = auc(fpr, tpr)

    # 3. 파일 저장 경로 조립 (스마트 이름표 규칙 준수)
    png_path = os.path.join(final_exp_dir, f"{time_prefix}_{split_name}_roc_curve{config_part}.png")
    csv_path = os.path.join(final_exp_dir, f"{time_prefix}_{split_name}_roc_data{config_part}.csv")

    # 4. 📈 [디자이너의 역할] ROC 커브 그래프 그리고 이미지 파일로 저장
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # 기준선(무작위 예측)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(f"ROC Curve - {split_name.upper()}")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # 여백 없이 깔끔하게 저장 후 메모리 닫기
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    # 5. 📄 [기록관의 역할] 엑셀이나 판다스에서 바로 열어볼 수 있게 수치 데이터를 CSV로 저장
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["False Positive Rate (FPR)", "True Positive Rate (TPR)", "Threshold"])
        for f_val, t_val, th_val in zip(fpr, tpr, thresholds):
            writer.writerow([f_val, t_val, th_val])

    print(f" [{split_name.upper()}] ROC Curve 이미지 저장 완료: {png_path}")
    print(f" [{split_name.upper()}] ROC 수치 데이터 CSV 저장 완료: {csv_path}")

    return {"png": png_path, "csv": csv_path}

def save_learning_curves(
    history: dict,
    final_exp_dir: str,
    time_prefix: str,
    config_part: str
) -> str:
    """
    Train/Val의 Epoch별 Loss 및 Accuracy 추이 그래프(Learning Curves)를 그려 저장합니다.
    """
    png_path = os.path.join(final_exp_dir, f"{time_prefix}_learning_curves{config_part}.png")
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 1. Loss 커브 서브플롯
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss", lw=2)
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss", lw=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # 2. Accuracy 커브 서브플롯
    plt.subplot(1, 2, 2)
    if "train_acc" in history and "val_acc" in history:
        plt.plot(epochs, history["train_acc"], "b-", label="Train Acc", lw=2)
        plt.plot(epochs, history["val_acc"], "r-", label="Val Acc", lw=2)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    
    return png_path