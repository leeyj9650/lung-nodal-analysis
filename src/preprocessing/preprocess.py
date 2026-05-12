# 학습 데이터셋을 실제로 만드는 단계
# 1. 원본 npy 수집 : 
#  data/raw/lidc-idri/slices/

# 2. label_policy 적용 : 
#  - benign / mailgnat결정
#  - ambiguous 제거

# 3. processed dataset 생성 : 
#  data/processed/method_a/npy/
#  data/processed/method_a/labels.csv


from pathlib import Path
import shutil
import pandas as pd
from tqdm import tqdm

from src.configs.config import *
from src.preprocessing.label_policy import get_label_policy


# =========================================================
# 기존 processed 데이터 삭제 + 초기화
# =========================================================
def reset_processed_dir(image_dir, labels_path):

    # 기존 npy 폴더 삭제
    if image_dir.exists():
        shutil.rmtree(image_dir)

    # 다시 생성
    image_dir.mkdir(parents=True, exist_ok=True)

    # labels.csv 삭제
    if labels_path.exists():
        labels_path.unlink()


# =========================================================
# Dataset 생성 함수
# =========================================================
def build_dataset(method_name, image_dir, labels_path):

    print(f"\n===== METHOD {method_name.upper()} START =====")

    # label policy 가져오기 (a or c)
    label_func = get_label_policy(method_name)

    # 원본 데이터 위치
    source_root = RAW_DATA_DIR / "slices"

    # 초기화
    reset_processed_dir(image_dir, labels_path)

    # npy 파일 수집
    npy_files = sorted(source_root.glob("LIDC-IDRI-*/*.npy"))

    print(f"Total files: {len(npy_files)}")

    records = []

    # =========================================================
    # 각 npy 파일 처리
    # =========================================================
    for file_path in tqdm(npy_files):

        patient_id = file_path.parent.name

        # 파일명에서 score 추출 (slice_XXX_score.npy)
        try:
            score = int(file_path.stem.split("_")[-1])
        except:
            continue

        # label 생성
        label = label_func(score)

        # 제외 대상이면 skip
        if label is None:
            continue

        # 새로운 파일 이름
        new_filename = f"{patient_id}_{file_path.name}"

        target_path = image_dir / new_filename

        # npy 복사
        shutil.copy2(file_path, target_path)

        # metadata 저장
        records.append({
            "patient_id": patient_id,
            "filename": new_filename,
            "original_path": str(file_path),
            "score": score,
            "label": label
        })

    # =========================================================
    # CSV 저장
    # =========================================================
    df = pd.DataFrame(records)
    df.to_csv(labels_path, index=False)

    print(f"\nDone METHOD {method_name.upper()}")
    print(f"Samples: {len(df)}")
    print("\nLabel distribution:")
    print(df["label"].value_counts())


# =========================================================
# 실행 테스트 → python -m src.preprocessing.preprocess
# =========================================================
if __name__ == "__main__":

    # Method A
    build_dataset(
        "a",
        METHOD_A_IMAGE_DIR,
        METHOD_A_LABELS
    )

    # Method C
    build_dataset(
        "c",
        METHOD_C_IMAGE_DIR,
        METHOD_C_LABELS
    )

    print("\nPREPROCESS COMPLETE")