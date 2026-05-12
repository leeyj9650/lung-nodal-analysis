# npy → 모델이 먹을 수 있는 Tensor로 변환


import numpy as np
import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from src.configs.config import *


# =========================================================
# HU Windowing (CT 값 제한)
# =========================================================
def apply_window(image):

    # HU 범위 설정
    lower = WINDOW_CENTER - WINDOW_WIDTH // 2
    upper = WINDOW_CENTER + WINDOW_WIDTH // 2

    # 범위 밖 값 제거
    image = np.clip(image, lower, upper)

    return image


# =========================================================
# Normalize (0~1 scaling)
# =========================================================
def normalize(image):

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image.astype(np.float32)


# =========================================================
# Center Crop
# =========================================================
def center_crop(image, size):

    h, w = image.shape

    cy, cx = h // 2, w // 2
    half = size // 2

    y1 = max(cy - half, 0)
    y2 = min(cy + half, h)
    x1 = max(cx - half, 0)
    x2 = min(cx + half, w)

    return image[y1:y2, x1:x2]


# =========================================================
# Dataset Class
# =========================================================
class LungDataset(Dataset):

    def __init__(self, df, image_dir):

        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = self.image_dir / row["filename"]

        # npy load
        image = np.load(path)

        # 1️⃣ HU window
        image = apply_window(image)

        # 2️⃣ center crop
        image = center_crop(image, CROP_SIZE)

        # 3️⃣ normalize
        image = normalize(image)

        # 4️⃣ resize (ResNet input)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 5️⃣ 3-channel 변환 (ResNet용)
        image = np.stack([image, image, image], axis=0)

        image = torch.tensor(image, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.long)

        return image, label


# =========================================================
# Patient-wise split (중요)
# =========================================================
def split_dataset(df):

    patients = df["patient_id"].unique()

    train_patients, temp_patients = train_test_split(
        patients,
        test_size=0.3,
        random_state=RANDOM_SEED
    )

    val_patients, test_patients = train_test_split(
        temp_patients,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

    train_df = df[df["patient_id"].isin(train_patients)]
    val_df = df[df["patient_id"].isin(val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]

    return train_df, val_df, test_df


# =========================================================
# DataLoader 생성
# =========================================================
def get_dataloaders(method="a", batch_size=BATCH_SIZE):

    if method == "a":
        df = pd.read_csv(METHOD_A_LABELS)
        image_dir = METHOD_A_IMAGE_DIR

    elif method == "c":
        df = pd.read_csv(METHOD_C_LABELS)
        image_dir = METHOD_C_IMAGE_DIR

    else:
        raise ValueError("method must be a or c")

    train_df, val_df, test_df = split_dataset(df)

    train_ds = LungDataset(train_df, image_dir)
    val_ds   = LungDataset(val_df, image_dir)
    test_ds  = LungDataset(test_df, image_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# =========================================================
# 실행 테스트 → python -m src.datasets.dataset
# =========================================================
if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders("a")

    images, labels = next(iter(train_loader))

    print("image shape:", images.shape)
    print("label shape:", labels.shape)