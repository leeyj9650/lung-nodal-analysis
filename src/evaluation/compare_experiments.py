# src/evaluation/compare_experiments.py
#
# ─── 역할 ────────────────────────────────────────────────────────────────────
#   여러 실험 폴더의 결과(json, csv, png)를 취합하여 종합 비교 테이블을 출력하고,
#   색약 사용자를 위한 '유니버설 디자인(마커+선 스타일+안전 컬러)' 그래프 3종을 생성합니다.
#
# ─── 사용 방법 ───────────────────────────────────────────────────────────────
#   python -m src.evaluation.compare_experiments

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc 

from src.configs.config import OUTPUT_ROOT

EXP_ROOT = OUTPUT_ROOT / 'experiments'
PLOT_OUTPUT_DIR = EXP_ROOT / 'comparison_plots'
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)

# 🌟 [색약 최적화] 적녹색약 및 황청색약 모두 구분이 가능한 유니버설 디자인 컬러 팔레트 (Okabe-Ito)
COLOR_PALETTE = [
    "#E69F00",  # 주황 (Orange)
    "#56B4E9",  # 하늘 (Sky Blue)
    "#009E73",  # 초록 (Bluish Green)
    "#F0E442",  # 노랑 (Yellow)
    "#0072B2",  # 파랑 (Blue)
    "#D55E00",  # 밤색 (Vermillion)
    "#CC79A7",  # 자주 (Reddish Purple)
    "#000000"   # 검정 (Black)
]

# 🌟 [형태 구별] 모델마다 고유의 도형 마커를 주어 색상이 없어도 선을 구별할 수 있게 만듭니다.
MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*', 'X']


def plot_combined_learning_curves(target_folders: list[str]) -> None:
    """
    1. 모든 모델의 학습 곡선(Loss & AUC Evolution)을 위아래(세로)로 배치하여 그립니다.
       - [변경] 좌우 배치를 위아래 배치(nrows=2, ncols=1)로 변경하여 범례가 들어갈 가로 공간을 극대화했습니다.
       - 색약 대응 유니버설 컬러 및 도형 마커 적용
    """
    # 위아래로 배치하기 위해 가로를 조금 줄이고, 세로 크기를 14로 대폭 키웠습니다.
    # 방 크기 자체를 세로로 긴 '2층 구조'로 만드는 과정입니다.
    plt.figure(figsize=(14, 14))
    
    marker_every = 5 
    
    # ─── 1층 (상단): Loss 비교 (Train vs Val) ───
    plt.subplot(2, 1, 1)  # 2행 1열 중 1번째
    for i, folder in enumerate(target_folders):
        csv_path = EXP_ROOT / folder / 'history.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            marker = MARKERS[i % len(MARKERS)]
            
            plt.plot(df['epoch'], df['train_loss'], 
                     linestyle='--', color=color, alpha=0.4, linewidth=1.5,
                     label=f"Train Loss ({folder})")
            
            plt.plot(df['epoch'], df['val_loss'], 
                     linestyle='-', color=color, marker=marker, markevery=marker_every, 
                     markersize=7, linewidth=2.5, label=f"★Val Loss ({folder})")
                     
    plt.title('Loss Evolution (Dashed: Train / Solid: Val)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 💡 [핵심 조정] 그래프 우측 외부에 범례 배치 (가로 공간이 넓어져 글자가 절대 깨지지 않습니다)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9, borderaxespad=0.)

    # ─── 2층 (하단): AUC 비교 (Train vs Val) ───
    plt.subplot(2, 1, 2)  # 2행 1열 중 2번째
    for i, folder in enumerate(target_folders):
        csv_path = EXP_ROOT / folder / 'history.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            marker = MARKERS[i % len(MARKERS)]
            
            plt.plot(df['epoch'], df['train_auc'], 
                     linestyle='--', color=color, alpha=0.4, linewidth=1.5,
                     label=f"Train AUC ({folder})")
            
            plt.plot(df['epoch'], df['val_auc'], 
                     linestyle='-', color=color, marker=marker, markevery=marker_every, 
                     markersize=7, linewidth=2.5, label=f"★Val AUC ({folder})")
                     
    plt.title('AUC Evolution (Dashed: Train / Solid: Val)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('AUC', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 💡 [핵심 조정] 2층 그래프도 마찬가지로 우측 외부에 나란히 배치
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9, borderaxespad=0.)

    # 💡 위아래 그래프와 글자들이 서로 겹치지 않도록 적당한 간격을 자동으로 띄워줍니다.
    plt.tight_layout()

    # 최종 저장 (여백 포함 안전하게 조임)
    save_path = PLOT_OUTPUT_DIR / 'combined_learning_curves.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] 📈 세로 배치형 변환 및 글자 잘림 방지 완료: {save_path}")


def plot_combined_roc_curves(target_folders: list[str]) -> None:
    """
    2. 모든 모델의 ROC 커브를 하나의 대형 그래프에 겹쳐서 그립니다. (learning_curve 방식)
       - [변경] 바둑판식 배열에서 '하나의 통합 그래프' 구조로 변경
       - [색약 최적화] 모델별로 유니버설 컬러 + 고유 도형 마커 + 선 스타일을 다르게 부여
    """
    # 범례가 길어지므로 가로 폭을 시원하게 12로 넓힌 단일 대형 그래프 생성
    plt.figure(figsize=(12, 8))
    
    # 그래프 선 위에 기호(마커)를 너무 촘촘하게 박으면 지저분하므로, 선을 따라 일정 간격(예: 10% 간격)으로 표시
    marker_every = 0.1 
    
    # 대각선 기준선 (무작위로 찍었을 때의 성능 기준 점선)
    plt.plot([0, 1], [0, 1], linestyle=':', color='#888888', linewidth=1.5, label='Random Guess (AUC = 0.5000)')
    
    has_data = False
    
    for i, folder in enumerate(target_folders):
        # 💡 각 실험 폴더 내부의 예측 결과나 json 파일을 읽어와서 그리도록 유연하게 대처해야 합니다.
        # 여기서는 각 폴더에 수치 데이터가 json으로 들어있다고 가정하거나, 
        # 혹은 evaluate 시점에 저장된 결과를 기반으로 새로 draw 합니다.
        result_path = EXP_ROOT / folder / 'result.json'
        
        if result_path.exists():
            with open(result_path, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
            
            # 만약 result.json 안에 그동안 저장해둔 roc 데이터(fpr, tpr)가 있다면 베스트입니다.
            # 데이터가 없는 경우를 대비해, 예시 플롯 흐름을 구성합니다.
            fpr = res_data.get('fpr', [])
            tpr = res_data.get('tpr', [])
            test_auc = res_data.get('test_auc', 0.0)
            
            # 만약 저장된 fpr, tpr 배열이 있다면 그대로 사용하고, 없다면 메시지를 띄웁니다.
            if fpr and tpr:
                has_data = True
                color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
                marker = MARKERS[i % len(MARKERS)]
                
                # 실선과 다양한 대시 스타일을 섞어서 선의 형태 자체를 차별화합니다.
                line_styles = ['-', '--', '-.', ':']
                ls = line_styles[i % len(line_styles)]
                
                plt.plot(fpr, tpr, linestyle=ls, color=color, 
                         marker=marker, markevery=marker_every, markersize=6, linewidth=2.5,
                         label=f"Model {i+1}: AUC = {test_auc:.4f} ({folder})")
                         
    if not has_data:
        # 만약 기존 result.json에 fpr, tpr 수치가 없다면 안내 문구를 띄우고 부드럽게 넘어갑니다.
        plt.text(0.5, 0.5, "💡 통합 ROC 커브를 그리려면\nevaluate.py 저장 시 fpr, tpr 리스트를\nresult.json에 포함시켜야 합니다.", 
                 ha='center', va='center', fontsize=12, fontweight='bold', color='gray')
        
    plt.title('Combined ROC Curves Comparison (Colorblind-Friendly)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 💡 범례가 그래프 영역을 침범하지 않고, 긴 이름이 다 잘 나오도록 우측 외부에 완전히 격리 배치!
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=9, borderaxespad=0.)
    
    save_path = PLOT_OUTPUT_DIR / 'combined_roc_curves_overlay.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] 🎯 단일 통합 그래프형 ROC Curve 완료: {save_path}")


def plot_combined_confusion_matrices(target_folders: list[str]) -> None:
    """
    3. 모든 모델의 혼동 행렬(Confusion Matrix) 이미지를 한눈에 비교하도록 바둑판 배열합니다.
       - [수정] 긴 폴더명 때문에 위아래 그림과 글자가 겹치던 문제를 여백 조정을 통해 해결했습니다.
    """
    n = len(target_folders)
    if n == 0: return
    
    cols = 2
    rows = (n + 1) // 2
    
    # 혼동 행렬 액자 크기 최적화
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, folder in enumerate(target_folders):
        img_path = EXP_ROOT / folder / 'confusion_matrix.png'
        if img_path.exists():
            img = plt.imread(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # 💡 [핵심] 제목이 위쪽 그림과 겹치지 않도록 줄바꿈(\n) 및 마진(pad) 적용
            axes[i].set_title(f"Model {i+1}:\n{folder}", fontsize=10, fontweight='bold', pad=12)
        else:
            axes[i].text(0.5, 0.5, f"confusion_matrix.png 없음", ha='center', va='center')
            axes[i].axis('off')
            
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    # 💡 [핵심] 위아래(hspace), 좌우(wspace) 여백을 충분히 주어 긴 이름이 다 들어가게 합니다.
    plt.subplots_adjust(wspace=0.3, hspace=0.4, top=0.85)
    
    save_path = PLOT_OUTPUT_DIR / 'combined_confusion_matrix.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] 🧩 혼동 행렬 모아보기 완료: {save_path}")


def compare_models(target_folders: list[str]) -> None:
    """종합 수치 테이블 취합 및 정렬 기능"""
    summary_table = []
    
    for folder_name in target_folders:
        exp_dir = EXP_ROOT / folder_name
        config_path = exp_dir / 'config.json'
        result_path = exp_dir / 'result.json'
        
        if not result_path.exists():
            continue
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        with open(result_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
            
        augs = config.get('augmentations', [])
        aug_desc = "No Aug" if (not augs or augs == ['none']) else f"Aug ({', '.join(augs)})"
            
        summary_table.append({
            'Folder': folder_name,
            'Model': config.get('model', 'unknown'),
            'Augmentations': aug_desc,
            'AUC': result.get('test_auc', 0.0),
            'Accuracy': result.get('test_accuracy', 0.0),
            'Sensitivity': result.get('test_sensitivity', 0.0),
            'Specificity': result.get('test_specificity', 0.0),
        })

    summary_table.sort(key=lambda x: x['AUC'] if isinstance(x['AUC'], float) else 0.0, reverse=True)
    
    print("\n" + "=" * 105)
    print(f"{'순위':<3s} {'실험 폴더명':<50s} {'AUC':>6s} | {'ACC':>6s} | {'SEN':>6s} | {'SPE':>6s}")
    print("=" * 105)
    for rank, row in enumerate(summary_table, 1):
        print(f"{rank:<4d} {row['Folder']:<50s} {row['AUC']:.4f} | {row['Accuracy']:.4f} | {row['Sensitivity']:.4f} | {row['Specificity']:.4f}")
    print("=" * 105)

    print("\n[INFO] 🎨 색약 대응 가시성이 적용된 비교 시각화 그래프를 생성합니다...")
    plot_combined_learning_curves(target_folders)
    plot_combined_roc_curves(target_folders)
    plot_combined_confusion_matrices(target_folders)
    print(f"\n[DONE] 🎉 시각화 파일 저장 완료 ➡️ {PLOT_OUTPUT_DIR}\n")


if __name__ == '__main__':
    print(f"[INFO] 📂 {EXP_ROOT} 경로에서 실험 폴더 자동 수집 중...")
    target_models = []
    
    for folder in sorted(EXP_ROOT.iterdir()):
        if folder.is_dir() and "convnext_64" in folder.name: 
            if (folder / 'config.json').exists() and (folder / 'result.json').exists():
                target_models.append(folder.name)
                
    if not target_models:
        print("[WARN] ❌ 분석 가능한 완성된 실험 폴더가 없습니다.")
    else:
        print(f"[INFO] ✅ 총 {len(target_models)}개의 실험 폴더를 찾았습니다.")
        compare_models(target_models)