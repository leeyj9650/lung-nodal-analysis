import argparse
import csv
import json
from pathlib import Path

# 📌 고정 default 경로 및 CSV 저장 루트 경로 상수로 지정
DEFAULT_BASE_DIR = "/home/lyj/Projects/2.5pj/outputs/experiments/260615_1111_dual_convnext_64+96_ep50_no_aug"
CSV_OUTPUT_ROOT = Path("/home/lyj/Projects/2.5pj/outputs/experiments")


def load_metrics(exp_dir_path: Path):
    """실험 폴더에서 history.csv와 result.json을 읽어 필요한 수치를 딕셔너리로 반환합니다."""
    history_csv = exp_dir_path / "history.csv"
    result_json = exp_dir_path / "result.json"

    if not history_csv.exists() or not result_json.exists():
        return None

    # 1. history.csv 분석
    history_data = []
    with open(history_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history_data.append(row)

    best_row = max(history_data, key=lambda x: float(x["val_auc"]))
    final_row = history_data[-1]

    # 2. result.json 분석
    with open(result_json, "r", encoding="utf-8") as f:
        res = json.load(f)

    subgroup = res.get("subgroup_auc", {})

    # 💡 n_nodules 집계 기준 실제 총 benign(193), malignant(94) 개수를 기반으로 정확히 역산합니다.
    sens = float(res.get("test_sensitivity", 0.0))
    spec = float(res.get("test_specificity", 0.0))
    
    n_malignant_total = 94  # 8 + 11 + 75
    n_benign_total = 193    # 76 + 72 + 45

    metrics = {
        "best_val_auc": float(best_row["val_auc"]),
        "final_val_auc": float(final_row["val_auc"]),
        "final_train_auc": float(final_row["train_auc"]),
        "auc_gap": float(final_row["train_auc"]) - float(final_row["val_auc"]),
        "val_loss": float(best_row["val_loss"]),  # Best Epoch 기준 val_loss
        "test_auc": float(res.get("test_auc", 0.0)),
        "test_accuracy": float(res.get("test_accuracy", 0.0)),
        "test_sensitivity": sens,
        "test_specificity": spec,
        "small_auc": subgroup.get("small", {}).get("auc", "N/A"),
        "inter_auc": subgroup.get("intermediate", {}).get("auc", "N/A"),
        "large_auc": subgroup.get("large", {}).get("auc", "N/A"),
        
        # 혼동 행렬 4대 요소 정수형 저장
        "cm_tp": round(n_malignant_total * sens),
        "cm_fn": round(n_malignant_total * (1 - sens)),
        "cm_fp": round(n_benign_total * (1 - spec)),
        "cm_tn": round(n_benign_total * spec),
    }
    return metrics


def fmt_with_delta(current_val, base_val):
    """현재 수치와 기준 수치를 비교하여 포맷팅합니다. 정수(개수)와 소수점을 판별하여 이쁘게 출력합니다."""
    if current_val == "N/A" or current_val is None:
        return "N/A"

    current_float = float(current_val)
    
    # 정수(개수 데이터)인지 판단 플래그
    is_int = current_float.is_integer() and (base_val == "N/A" or base_val is None or float(base_val).is_integer())

    if base_val == "N/A" or base_val is None:
        return f"{int(current_float)}" if is_int else f"{current_float:.4f}"

    base_float = float(base_val)
    delta = current_float - base_float

    if is_int:
        c_str = f"{int(current_float)}"
        d_str = f"+{int(delta)}" if delta > 0 else f"{int(delta)}"
        return f"{c_str} ({d_str})"
    else:
        d_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        return f"{current_float:.4f} ({d_str})"


def extract_and_save_csv(exp_dir_path: str, base_dir_path: str | None = None):
    exp_dir = Path(exp_dir_path)
    
    if not exp_dir.exists():
        exp_dir = CSV_OUTPUT_ROOT / exp_dir_path

    current_metrics = load_metrics(exp_dir)

    if not current_metrics:
        print(f"[오류] 현재 실험 폴더({exp_dir})의 로그 파일들을 읽을 수 없습니다.")
        return

    # 대조할 기준 폴더 설정
    base_metrics = None
    target_base_path = base_dir_path if base_dir_path else DEFAULT_BASE_DIR

    if target_base_path:
        base_dir = Path(target_base_path)
        base_metrics = load_metrics(base_dir)
        if not base_metrics:
            print(f"[경고] 기준(default) 폴더의 로그를 읽을 수 없어 증감 계산을 제외합니다.")

    def get_base_v(key):
        return base_metrics[key] if base_metrics else "N/A"

    # 3. 기본 메트릭 데이터 매핑
    rows_to_save = [
        {"평가 항목": "best val AUC", "key": "best_val_auc"},
        {"평가 항목": "최종 val AUC", "key": "final_val_auc"},
        {"평가 항목": "최종 train AUC", "key": "final_train_auc"},
        {"평가 항목": "AUC gap", "key": "auc_gap"},
        {"평가 항목": "val loss (at Best Epoch)", "key": "val_loss"},
        {"평가 항목": "테스트 결과 AUC", "key": "test_auc"},
        {"평가 항목": "Accuracy", "key": "test_accuracy"},
        {"평가 항목": "Sensitivity", "key": "test_sensitivity"},
        {"평가 항목": "Specificity", "key": "test_specificity"},
        {"평가 항목": "Subgroup AUC (small)", "key": "small_auc"},
        {"평가 항목": "Subgroup AUC (intermediate)", "key": "inter_auc"},
        {"평가 항목": "Subgroup AUC (large)", "key": "large_auc"},
    ]

    # 4. 저장할 파일 경로 설정 (분석 폴더 내부에 자동 안착)
    output_file_path = exp_dir / f"{exp_dir.name}.csv"

    # 5. CSV 파일 쓰기
    with open(output_file_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        # 헤더 기록
        writer.writerow(["평가 항목", "실험 결과 수치 (증감)", ""])
        
        # 본문 내용 기록
        for row in rows_to_save:
            k = row["key"]
            formatted_val = fmt_with_delta(current_metrics[k], get_base_v(k))
            writer.writerow([row["평가 항목"], formatted_val])
        
        # 🌟 요구사항 1: Subgroup AUC 이후 한 칸 띄우기
        writer.writerow([])
        
        # 🌟 요구사항 2: 매트릭스 형태로 예쁘게 정렬하여 쓰기
        # 행 1: TN (정상 적중), FP (정상 오진)
        # 행 2: FN (악성 놓침), TP (악성 적중)
        tn_val = fmt_with_delta(current_metrics["cm_tn"], get_base_v("cm_tn"))
        fp_val = fmt_with_delta(current_metrics["cm_fp"], get_base_v("cm_fp"))
        fn_val = fmt_with_delta(current_metrics["cm_fn"], get_base_v("cm_fn"))
        tp_val = fmt_with_delta(current_metrics["cm_tp"], get_base_v("cm_tp"))
        
        writer.writerow([tn_val, fp_val])
        writer.writerow([fn_val, tp_val])

    print("\n" + "=" * 65)
    print(f"✅ CSV 저장 완료!")
    print(f"  - 저장 경로: {output_file_path}")
    print(f"  - 대조군(Default): {Path(target_base_path).name if base_metrics else '없음'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="결과를 추출할 현재 실험 폴더 경로를 지정하세요.",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default=None,
        help="기준이 되는 default 실험 폴더 경로를 지정하세요.",
    )
    args = parser.parse_args()

    extract_and_save_csv(args.exp_dir, args.base_dir)