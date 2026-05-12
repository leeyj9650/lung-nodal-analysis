# 프로젝트 전체 설정 관리 파일
# 데이터 경로, batch size, learning rate, image size, output 경로 관리

from pathlib import Path


# =========================================================
# 프로젝트 최상위 루트 경로
# =========================================================
# 현재 파일 위치:
# src/configs/config.py
#
# parent         -> configs/
# parent.parent  -> src/
# parent.parent.parent -> Project2/
#
# 즉 Project2 루트를 자동으로 찾음
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# =========================================================
# Data Directory
# =========================================================
# 원본 데이터 / 전처리 데이터 저장 위치
# =========================================================
DATA_DIR = PROJECT_ROOT / "data"

# 원본 LIDC 데이터
RAW_DATA_DIR = DATA_DIR / "raw" / "lidc-idri"

# 전처리 결과 저장 폴더
PROCESSED_DIR = DATA_DIR / "processed"


# =========================================================
# Method A Dataset
# =========================================================
# Method A:
# 1,2 -> benign
# 4,5 -> malignant
# 3 제외
# =========================================================
METHOD_A_DIR = PROCESSED_DIR / "method_a"

# npy 저장 위치
METHOD_A_IMAGE_DIR = METHOD_A_DIR / "npy"

# labels.csv 위치
METHOD_A_LABELS = METHOD_A_DIR / "labels.csv"


# =========================================================
# Method C Dataset
# =========================================================
# Method C:
# 1 -> benign
# 5 -> malignant
# 2,3,4 제외
# =========================================================
METHOD_C_DIR = PROCESSED_DIR / "method_c"

# npy 저장 위치
METHOD_C_IMAGE_DIR = METHOD_C_DIR / "npy"

# labels.csv 위치
METHOD_C_LABELS = METHOD_C_DIR / "labels.csv"


# =========================================================
# Output Directory
# =========================================================
# 학습 결과 저장 위치
# =========================================================
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# 학습된 모델(.pth)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

# 학습 로그(json)
LOG_DIR = OUTPUT_DIR / "logs"

# ROC curve, confusion matrix 등
FIGURE_DIR = OUTPUT_DIR / "figures"

# Grad-CAM 이미지 저장
GRADCAM_DIR = OUTPUT_DIR / "gradcam"

# 예측 결과 저장(csv)
PREDICTION_DIR = OUTPUT_DIR / "predictions"


# =========================================================
# Image Settings
# =========================================================

# 최종 입력 이미지 크기
IMAGE_SIZE = 224

# Center Crop 크기
CROP_SIZE = 128

# CT HU Window 설정
WINDOW_CENTER = -600
WINDOW_WIDTH = 1500

# HU clipping 범위
HU_MIN = -1000
HU_MAX = 400


# =========================================================
# Training Settings
# =========================================================

# Batch Size
BATCH_SIZE = 16

# Epoch 수
NUM_EPOCHS = 30

# Learning Rate
LEARNING_RATE = 1e-4

# Weight Decay (L2 Regularization)
WEIGHT_DECAY = 1e-5

# DataLoader Worker 수
NUM_WORKERS = 4

# Random Seed
RANDOM_SEED = 42


# =========================================================
# Model Settings
# =========================================================

# 사용할 모델 이름
MODEL_NAME = "resnet50"

# ImageNet pretrained 사용 여부
PRETRAINED = True


# =========================================================
# Class Names
# =========================================================
CLASS_NAMES = {
    0: "benign",
    1: "malignant"
}


# =========================================================
# 자동 폴더 생성
# =========================================================
# 프로젝트 실행 시 필요한 폴더 자동 생성
# =========================================================
for path in [

    # Method A
    METHOD_A_IMAGE_DIR,

    # Method C
    METHOD_C_IMAGE_DIR,

    # Outputs
    CHECKPOINT_DIR,
    LOG_DIR,
    FIGURE_DIR,
    GRADCAM_DIR,
    PREDICTION_DIR

]:
    path.mkdir(
        parents=True,
        exist_ok=True
    )

# =========================================================
# 실행 테스트 → 터미널에서 python -m src.configs.config
# =========================================================
if __name__ == "__main__":

    print("===== CONFIG TEST =====")

    print()

    print("PROJECT_ROOT:")
    print(PROJECT_ROOT)

    print()

    print("RAW_DATA_DIR:")
    print(RAW_DATA_DIR)

    print()

    print("METHOD_A_LABELS:")
    print(METHOD_A_LABELS)

    print()

    print("CHECKPOINT_DIR:")
    print(CHECKPOINT_DIR)

    print()

    print("IMAGE_SIZE:", IMAGE_SIZE)

    print("BATCH_SIZE:", BATCH_SIZE)

    print()

    print("Config Load Success")