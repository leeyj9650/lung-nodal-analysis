"""
src/utils/file_naming.py

역할:
- 학습 코드에서 호출하는 저장 관련 전담 함수 모음
- 스마트 이름표 계산 및 타임스탬프 기반 중복 방지 로직 포함
- 원본 코드의 함수명과 구조를 100% 그대로 유지
"""

import os
import csv
import json
import sys
from datetime import datetime
from typing import List
import numpy as np


def generate_config_suffix(args) -> str:
    """
    사용자가 터미널(sys.argv)에 실제로 입력한 옵션만 분석하여 
    직관적이고 깔끔한 폴더명/이름표를 생성합니다.
    """
    # 1. 사용자가 터미널에 직접 입력한 '--파라미터' 목록만 추출
    user_changed_args = [arg.lstrip('-') for arg in sys.argv if arg.startswith('--')]
    
    # 2. crop_mode 결정
    is_dynamic = "dynamic_crop" in user_changed_args or getattr(args, "crop_mode", "fixed") == "dynamic"
    crop_mode = "dynamic" if is_dynamic else "fixed"
    
    suffix_parts = []
    
    # 3. 사용자가 터미널에 '명시적으로 입력한 인자'가 있을 때만 이름표에 추가
    # 단, 경로 관련 설정이나 이미 crop_mode로 반영된 인자는 이름표에서 제외합니다.
    ignore_args = ['log_dir', 'pred_dir', 'save_dir', 'data_dir', 'split_json', 'exp_name', 'crop_mode', 'dynamic_crop']
    
    for arg_name in user_changed_args:
        if arg_name in ignore_args:
            continue
        if hasattr(args, arg_name):
            val = getattr(args, arg_name)
            # 깔끔한 매핑을 위해 'image_size'는 'image'로 축약해 표현
            display_name = "image" if arg_name == "image_size" else arg_name
            suffix_parts.append(f"{display_name}{val}")
            
    # 4. 사용자가 추가 옵션을 입력했다면 뒤에 붙이고, 없으면 깔끔하게 crop_mode만 반환
    if suffix_parts:
        return f"_{crop_mode}_" + "_".join(suffix_parts)
    else:
        return f"_{crop_mode}"

def get_unique_path(base_path: str) -> str:
    """
    실험 폴더명(또는 기본 경로)이 이미 존재한다면 뒤에 _1, _2를 붙여 폴더 중복을 차단합니다.
    """
    if not os.path.exists(base_path):
        return base_path
        
    counter = 1
    while True:
        # 폴더명 뒤에 _1, _2를 붙여서 중복되지 않는 경로를 찾습니다.
        backup_path = f"{base_path}_{counter}"
        if not os.path.exists(backup_path):
            return backup_path
        counter += 1


def save_history(history_path: str, train_history: List[dict]):
    """매 에폭마다의 중간 학습 기록(Loss, ACC)을 CSV 파일로 저장합니다."""
    if not train_history:
        return
        
    fieldnames = list(train_history[0].keys())
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_history)


def save_test_metrics(
    metrics_path: str, 
    args, 
    test_loss: float, 
    test_metrics: dict, 
    best_epoch: int, 
    best_auc: float, 
    split_info: dict,
    roc_curve_path: str = None,      # 🌟 새로 추가
    roc_curve_csv_path: str = None   # 🌟 새로 추가
):
    """최종 모델의 테스트 성적표와 하이퍼파라미터 세팅을 JSON 파일로 저장합니다."""
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best_epoch,
        "best_val_auc": None if np.isnan(best_auc) else float(best_auc),
        "test_loss": float(test_loss),
        "test_metrics": {k: (None if isinstance(v, float) and np.isnan(v) else float(v)) for k, v in test_metrics.items()},
        "args": vars(args),
        "split": split_info,
        "roc_curve_path": roc_curve_path,         # 🌟 JSON 데이터에 경로 추가
        "roc_curve_csv_path": roc_curve_csv_path, # 🌟 JSON 데이터에 경로 추가
    }
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def save_test_predictions(
    pred_path: str,
    dataset,
    test_subset,
    test_true: list,
    test_pred: list,
    test_prob: list,
):
    """
    최고 성능 모델의 Test 예측 확률 및 정답, 결절 메타데이터를 CSV로 결합 보관합니다.
    """
    # 💡 1. 현재 로드된 데이터셋 폴더 경로명에 'dynamic'이 있는지 검사하여 확실하게 판별
    npy_root_str = str(getattr(dataset, "npy_root", "")).lower()
    is_dynamic = "dynamic" in npy_root_str

    rows = []
    for i, idx in enumerate(test_subset.indices):
        sample = dataset.samples[idx]
        sample_idx = sample.get("sample_idx", idx)
        
        # 결절의 실제 지름(diameter_mm) 추출
        diameter = sample.get("diameter_mm", 0.0)
        
        # 💡 2. 데이터가 다이내믹 모드일 때의 처리 규칙 정립
        if is_dynamic:
            # samples.json에 기록된 crop_mm가 있으면 쓰고, 없으면 (지름 * 4배) 규칙으로 역산 보완
            crop_mm_val = sample.get("crop_mm")
            if crop_mm_val is None:
                crop_mm_val = float(diameter * 4.0) if diameter > 0 else 64.0
            
            # 수치 데이터 저장을 위해 dynamic_crop_size 칸에도 수치(mm)를 동일하게 매핑
            dynamic_size_val = crop_mm_val
        else:
            # 고정 크기 모드일 때는 깔끔하게 64 고정
            crop_mm_val = 64.0
            dynamic_size_val = 64

        rows.append({
            "dataset_idx": idx,
            "sample_idx": sample_idx,
            "subject_id": sample.get("subject_id", ""),
            "series_uid": sample.get("series_uid", ""),
            "nodule_idx": sample.get("nodule_idx", ""),
            "true_label": test_true[i],
            "pred_label": test_pred[i],
            "prob_0": test_prob[i][0],
            "prob_1": test_prob[i][1],
            "diameter_mm": diameter,
            "volume_mm3": sample.get("volume_mm3", ""), 
            "size_group": sample.get("size_group", ""),  # 중복 제거 완료
            "crop_mm": float(crop_mm_val),
            "dynamic_crop_size": dynamic_size_val,
            "npy_path": sample.get("npy_path", ""),
        })

    fieldnames = [
        "dataset_idx", "sample_idx", "subject_id", "series_uid", "nodule_idx", 
        "true_label", "pred_label", "prob_0", "prob_1", 
        "diameter_mm", "volume_mm3", "size_group", 
        "crop_mm", "dynamic_crop_size", "npy_path"
    ]
    
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def get_experiment_id(config_suffix: str) -> str:
    """
    현재 시간(YYMMDDHHMM)과 하이퍼파라미터 접미사를 예쁘게 결합하여 
    이번 실험의 통합 폴더명이 될 '순수 실험 ID'를 발행합니다.
    예: 2606021550_fixed_crop_size64_image_size64
    """
    time_stamp = datetime.now().strftime("%y%m%d%H%M")
    
    # config_suffix가 언더바(_)로 시작하면 그대로 붙이고, 아니면 언더바를 사이에 넣어줍니다.
    if config_suffix.startswith("_"):
        return f"{time_stamp}{config_suffix}"
    return f"{time_stamp}_{config_suffix}"

def generate_npy_dir_name(args) -> str:
    """make_npy 실행 시 사용할 폴더명을 생성합니다. (예: fixed_crop64_img64)"""
    suffix = generate_config_suffix(args)
    return suffix.lstrip('_')