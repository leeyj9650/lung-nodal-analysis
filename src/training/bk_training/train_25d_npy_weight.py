"""
src/training/train_25d_npy_weight.py

역할
- LIDC-IDRI NPY cache 기반 weighted 2.5D 이진분류 학습 실행 코드 (GDN 모델 전용 2.5D 최종 변환 버전)
- 파일 저장 및 이름표 관련 로직은 최신 train_lidc_25d_npy_weight.py의 로직과 완벽히 동기화되었습니다.
- [수정 사항] 팀원이 생성한 '_25d' 폴더 구조를 자동으로 완벽하게 탐색하고 인식하도록 주소 체계를 수정했습니다.
"""

import os
import argparse
import random
import json
import glob  
from pathlib import Path
from typing import Tuple, List
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime

from src.datasets.lidc_npy_25d_dataset import LIDCNpy25DDataset
from src.models.ResNet_GDN import ResNet_GDN
from src.utils.metrics import compute_binary_metrics, print_binary_report
from src.configs.config import SPLIT_JSON
from src.utils.make_graph import save_roc_curve_and_csv, save_learning_curves  
from src.utils.data_augment import Medical25DAugment

from src.utils.file_naming import (
    get_unique_path,
    save_history,
    save_test_metrics,
    save_test_predictions,
    get_experiment_id
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_subset_labels(subset) -> List[int]:
    labels = []
    for idx in subset.indices:
        labels.append(int(subset.dataset.samples[idx]["label"]))
    return labels


def load_subject_split_json(split_json_path: str):
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(f"split.json 파일이 없습니다: {split_json_path}")
    with open(split_json_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)
    return {str(x) for x in split_dict["train"]}, {str(x) for x in split_dict["val"]}, {str(x) for x in split_dict["test"]}, split_dict.get("meta", {})


def make_subject_split_subsets(dataset, train_ids, val_ids, test_ids):
    train_indices, val_indices, test_indices = [], [], []
    missing_subject_ids = set()

    for idx, sample in enumerate(dataset.samples):
        subject_id = str(sample["subject_id"])
        if subject_id in train_ids:
            train_indices.append(idx)
        elif subject_id in val_ids:
            val_indices.append(idx)
        elif subject_id in test_ids:
            test_indices.append(idx)
        else:
            missing_subject_ids.add(subject_id)

    if not train_indices or not val_indices or not test_indices:
        raise ValueError("train/val/test 중 비어 있는 split이 있습니다.")

    validate_no_subject_leakage(dataset, train_indices, val_indices, test_indices)
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)


def validate_no_subject_leakage(dataset, train_indices, val_indices, test_indices):
    train_subjects = {str(dataset.samples[idx]["subject_id"]) for idx in train_indices}
    val_subjects = {str(dataset.samples[idx]["subject_id"]) for idx in val_indices}
    test_subjects = {str(dataset.samples[idx]["subject_id"]) for idx in test_indices}
    if (train_subjects & val_subjects) or (train_subjects & test_subjects) or (val_subjects & test_subjects):
        raise ValueError("Patient-level data leakage가 발견되었습니다.")
    print("✅ Patient-level data leakage 없음")


def print_distribution(name: str, labels: List[int]) -> Counter:
    counter = Counter(labels)
    print(f"{name} distribution: {counter}")
    return counter


def make_class_weights_from_train(train_labels: List[int], device: torch.device, num_classes: int = 2) -> torch.Tensor:
    counter = Counter(train_labels)
    total = len(train_labels)
    weights = []
    for class_idx in range(num_classes):
        count = counter.get(class_idx, 0)
        if count == 0:
            raise ValueError(f"Train set에 class {class_idx}가 없습니다.")
        weights.append(total / count)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_count += images.size(0)
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name="Val"):
    model.eval()
    total_loss, total_count = 0.0, 0
    all_true, all_prob, all_pred = [], [], []
    for images, labels in tqdm(loader, desc=split_name, leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        prob = torch.softmax(logits, dim=1)
        pred = torch.argmax(prob, dim=1)
        total_loss += loss.item() * images.size(0)
        total_count += images.size(0)
        all_true.extend(labels.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
    return total_loss / total_count, compute_binary_metrics(all_true, all_prob, all_pred), all_true, all_pred, all_prob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_root", type=str, default="data/processed/npy_cache")
    parser.add_argument("--cache", type=str, default=None, help="직접 지정할 하위 캐시 폴더명 (예: fixed_crop64_img128_25d)")
    parser.add_argument("--samples_json", type=str, default=None)
    parser.add_argument("--split_json", type=str, default=str(SPLIT_JSON))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--crop_size", type=int, default=32)
    parser.add_argument("--dynamic_crop", action="store_true", help="Enable dynamic cropping mode")
    parser.add_argument("--crop_mode", type=str, default="fixed", choices=["fixed", "dynamic", "both"])
    parser.add_argument("--crop_scale", type=float, default=4.0)
    parser.add_argument("--crop_min_mm", type=float, default=32.0)
    parser.add_argument("--crop_max_mm", type=float, default=96.0)
    parser.add_argument("--batch_size", type=int, default=16)  
    parser.add_argument("--epochs", type=int, default=50)     
    parser.add_argument("--ch_sz", type=int, default=16, help="모델의 기본 채널 크기")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--pred_dir", type=str, default=None)
    args = parser.parse_args()

    args.npy_root = str(Path(args.npy_root).resolve())
    args.split_json = str(Path(args.split_json).resolve())

    if args.dynamic_crop:
        args.crop_mode = "dynamic"

    # -----------------------------------------------------------------
    # 🌟 [2.5D 핵심 수정 구간] 스마트 경로 확정 및 자동 탐색 로직 개선
    # -----------------------------------------------------------------
    if args.cache is not None:
        args.npy_root = os.path.join(args.npy_root, args.cache)
        if "dynamic" in args.cache:
            args.crop_mode = "dynamic"
        elif "fixed" in args.cache:
            args.crop_mode = "fixed"
            
        if not os.path.exists(args.npy_root):
            raise FileNotFoundError(f"❌ 지정하신 캐시 폴더가 존재하지 않습니다: {args.npy_root}")
    else:
        search_pattern = os.path.join(args.npy_root, f"{args.crop_mode}_*_25d")
        matching_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
        
        # 만약 만에 하나 _25d 가 안 붙어있을 수도 있으니, 예외 대비용 범용 서치 한 번 더 레이어 구성
        if not matching_dirs:
            search_pattern_fallback = os.path.join(args.npy_root, f"{args.crop_mode}_*")
            matching_dirs = [d for d in glob.glob(search_pattern_fallback) if os.path.isdir(d) and "_25d" in d]

        if not matching_dirs:
            raise FileNotFoundError(
                f"❌ [{args.crop_mode}] 모드에 해당하는 2.5D 전처리 캐시 폴더(_25d)를 '{args.npy_root}'에서 찾을 수 없습니다.\n"
                f"데이터 전처리(make_npy.py)가 정상적으로 완료되었는지 폴더명을 확인해주세요."
            )
        # 가장 최근에 만들어진 2.5D 폴더를 선택합니다.
        matching_dirs.sort(key=os.path.getmtime, reverse=True)
        args.npy_root = matching_dirs[0]

    if args.crop_mode == "both" or "both" in os.path.basename(args.npy_root):
        raise ValueError(
            "학습(train) 시에는 'both' 모드를 지정하거나 사용할 수 없습니다.\n"
            "make_npy.py로 생성된 하위 폴더 중 하나를 선택해 주세요."
        )

    folder_name = os.path.basename(args.npy_root)
    config_suffix = f"_{folder_name}"

    print(f"▶ [알림] 실제로 데이터를 가져올 2.5D 경로: {args.npy_root}")
    print(f"▶ [알림] 자동 인식된 설정 이름표(config_suffix): {config_suffix}")

    args.save_dir = os.path.join(args.save_dir, args.crop_mode)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"체크포인트 저장 경로 확정: {args.save_dir}")

    time_stamp = datetime.now().strftime("%y%m%d%H%M")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 데이터셋 설정
    dataset = LIDCNpy25DDataset(npy_root=args.npy_root, samples_json=args.samples_json)
    print(f"Total samples: {len(dataset)}")
    
    train_ids, val_ids, test_ids, split_meta = load_subject_split_json(args.split_json)
    train_dataset, val_dataset, test_dataset = make_subject_split_subsets(dataset, train_ids, val_ids, test_ids)

    print("============================================================")
    print("Dataset split")
    print("============================================================")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples  : {len(val_dataset)}")
    print(f"Test samples : {len(test_dataset)}")
    print("============================================================")

    train_counter = print_distribution("Train", get_subset_labels(train_dataset))
    val_counter = print_distribution("Val", get_subset_labels(val_dataset))
    test_counter = print_distribution("Test", get_subset_labels(test_dataset))

    class_weights = make_class_weights_from_train(get_subset_labels(train_dataset), device)
    print(f"Train class weights: {class_weights}")

    augment_fn = Medical25DAugment(p=0.5) 
    
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, augment_func):
            self.subset = subset
            self.augment_func = augment_func
            
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            image = self.augment_func(image)  # 이미지 변형 적용
            return image, label
            
        def __len__(self):
            return len(self.subset)

    augmented_train_dataset = AugmentedDataset(train_dataset, augment_fn)
    print("Train dataset에 상하반전이 제외된 의료용 데이터 증강(Augmentation)이 적용되었습니다.")

    # DataLoader
    train_loader = DataLoader(
        augmented_train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, 
        pin_memory=True, worker_init_fn=seed_worker
        )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = ResNet_GDN(ch_sz=args.ch_sz, num_classes=2)
    model = model.to(device)

    model_name = model.__class__.__name__
    best_model_path = os.path.join(args.save_dir, f"{time_stamp}_best_{model_name}_25d_{config_suffix}.pt")

    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    split_info = {
        "train_size": len(train_dataset), "val_size": len(val_dataset), "test_size": len(test_dataset),
        "train_distribution": dict(train_counter), "val_distribution": dict(val_counter), "test_distribution": dict(test_counter),
        "class_weights": class_weights.detach().cpu().tolist(), "split_json": args.split_json, "split_meta": split_meta,
    }

    raw_history_path = os.path.join(args.save_dir, f"temp_train_history_{model_name}{config_suffix}.csv")
    best_auc, best_epoch = -1.0, -1
    train_history = []

    # 🔄 1. 학습 루프 진행
    for epoch in range(1, args.epochs + 1):
        train_loss_epoch, train_acc_epoch = train_one_epoch(model, train_loader, train_criterion, optimizer, device)
        val_loss_epoch, val_metrics_epoch, _, _, _ = evaluate(model, val_loader, eval_criterion, device, "Val")

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss_epoch:.4f}, "
            f"train_acc={train_acc_epoch:.4f} | "
            f"val_loss={val_loss_epoch:.4f}, "
            f"val_acc={val_metrics_epoch['acc']:.4f}, "
            f"val_macro_f1={val_metrics_epoch['macro_f1']:.4f}, "
            f"val_auc={val_metrics_epoch['auc']:.4f}"
        )

        train_history.append({
            "epoch": epoch, "train_loss": train_loss_epoch, "train_acc": train_acc_epoch,
            "val_loss": val_loss_epoch, "val_acc": val_metrics_epoch["acc"], "val_macro_f1": val_metrics_epoch["macro_f1"], "val_auc": val_metrics_epoch["auc"],
        })
        
        save_history(raw_history_path, train_history)

        current_auc = val_metrics_epoch["auc"]
        if not np.isnan(current_auc) and current_auc > best_auc:
            best_auc, best_epoch = current_auc, epoch
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "best_auc": best_auc, "args": vars(args), "split": split_info}, best_model_path)
            print(f"Best model saved: {best_model_path}")

    # 📂 2. 폴더 보관소 구축
    raw_exp_id = get_experiment_id(config_suffix)
    
    base_exp_dir = os.path.join("outputs", "logs_and_predictions", f"{model_name}_{raw_exp_id}")
    final_exp_dir = get_unique_path(base_exp_dir)
    os.makedirs(final_exp_dir, exist_ok=True)
    
    args.log_dir = final_exp_dir
    args.pred_dir = final_exp_dir

    experiment_id = os.path.basename(final_exp_dir)
    time_prefix = experiment_id[:10]         
    config_part = experiment_id[10:]         

    test_metrics_path = os.path.join(final_exp_dir, f"{time_prefix}_final_test_metrics{config_part}.json")
    test_predictions_path = os.path.join(final_exp_dir, f"{time_prefix}_test_predictions{config_part}.csv")
    history_path = os.path.join(final_exp_dir, f"{time_prefix}_train_history{config_part}.csv")
    
    if os.path.exists(raw_history_path):
        os.rename(raw_history_path, history_path)

    print(f"\nName : {experiment_id}")
    print("\nBest validation result")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val AUC: {best_auc:.4f}")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device)["model_state_dict"])
        print(f"Loaded best model: {best_model_path}")
    else:
        print("[WARN] best model 파일이 없습니다. 현재 마지막 epoch 모델로 test 평가합니다.")

    # 최종 평가 연산
    train_loss, train_metrics, _, _, _ = evaluate(model, train_loader, eval_criterion, device, "Train")
    val_loss, val_metrics, _, _, _ = evaluate(model, val_loader, eval_criterion, device, "Val")
    test_loss, test_metrics, test_true, test_pred, test_prob = evaluate(model, test_loader, eval_criterion, device, "Test")

    print("\nFinal train/val/test result")
    print(f"train_loss={train_loss:.4f}, train_acc={train_metrics['acc']:.4f}, train_macro_f1={train_metrics['macro_f1']:.4f}, train_auc={train_metrics['auc']:.4f}")
    print(f"val_loss={val_loss:.4f}, val_acc={val_metrics['acc']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}, val_auc={val_metrics['auc']:.4f}")
    print(f"test_loss={test_loss:.4f}, test_acc={test_metrics['acc']:.4f}, test_macro_f1={test_metrics['macro_f1']:.4f}, test_auc={test_metrics['auc']:.4f}")

    print_binary_report(test_true, test_pred)

    # 📈 ROC 그래프 자동 생성
    roc_paths = save_roc_curve_and_csv(
        all_true=test_true,
        all_prob=test_prob,
        final_exp_dir=final_exp_dir,      
        time_prefix=time_prefix,          
        config_part=config_part,          
        split_name="test"                 
    )

    # 💾 메트릭 및 예측값 저장
    save_test_metrics(
        metrics_path=test_metrics_path,
        args=args,
        test_loss=test_loss,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_auc=best_auc,
        split_info=split_info,
        roc_curve_path=roc_paths.get("png"),  
        roc_curve_csv_path=roc_paths.get("csv"),
    )

    save_test_predictions(
        pred_path=test_predictions_path,
        dataset=dataset,
        test_subset=test_dataset,
        test_true=test_true,
        test_pred=test_pred,
        test_prob=test_prob,
    )

    formatted_history = {
        "train_loss": [epoch_data["train_loss"] for epoch_data in train_history if "train_loss" in epoch_data],
        "val_loss": [epoch_data["val_loss"] for epoch_data in train_history if "val_loss" in epoch_data],
        "train_acc": [epoch_data["train_acc"] for epoch_data in train_history if "train_acc" in epoch_data],
        "val_acc": [epoch_data["val_acc"] for epoch_data in train_history if "val_acc" in epoch_data]
    }

    # 📈 러닝 커브 그래프 생성
    curves_path = save_learning_curves(
        history=formatted_history,      
        final_exp_dir=final_exp_dir,    
        time_prefix=time_prefix,        
        config_part=config_part         
    )

    print("\nSaved outputs (통합 보관 완료)")
    print(f"Best Model (.pt)    : {best_model_path}")
    print(f"Train history       : {history_path}")
    print(f"Learning curves     : {curves_path if curves_path else 'Saved in final_exp_dir'}")
    print(f"Final test metrics : {test_metrics_path}")
    print(f"Test predictions    : {test_predictions_path}")


if __name__ == "__main__":
    main()