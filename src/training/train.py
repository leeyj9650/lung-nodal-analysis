# 1. 전체 epoch loop
# 2. best model 저장
# 3. validation AUC 기준 선택
# 4. training log  저장


import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse

from src.configs.config import *
from src.datasets.dataset import get_dataloaders
from src.models.resnet import create_model
from src.training.engine import train_one_epoch, validate


# =========================================================
# device 설정
# =========================================================
def get_device():

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# main
# =========================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="a", choices=["a", "c"])
    args = parser.parse_args()

    method = args.method

    device = get_device()

    print(f"Device: {device}")
    print(f"Method: {method}")

    # -----------------------------------------------------
    # dataloader
    # -----------------------------------------------------
    train_loader, val_loader , _ = get_dataloaders(method)

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------
    model = create_model(pretrained=True)
    model = model.to(device)

    # -----------------------------------------------------
    # loss / optimizer
    # -----------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_auc = 0.0
    history = []

    # =====================================================
    # training loop
    # =====================================================
    for epoch in range(NUM_EPOCHS):

        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_auc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"train loss: {train_loss:.4f}")
        print(f"val loss: {val_loss:.4f}")
        print(f"val auc: {val_auc:.4f}")

        # -------------------------------------------------
        # best model 저장
        # -------------------------------------------------
        if val_auc > best_auc:

            best_auc = val_auc

            save_path = CHECKPOINT_DIR / f"best_model_{method}.pth"

            torch.save(model.state_dict(), save_path)

            print(f"BEST MODEL SAVED: {save_path}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc
        })

    # =====================================================
    # log 저장
    # =====================================================
    log_path = LOG_DIR / f"train_{method}.json"

    with open(log_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\nTRAINING COMPLETE")
    print(f"Best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    main()