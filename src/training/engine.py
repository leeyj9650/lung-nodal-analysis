# 1. train step :
#  forward → loss → backward → update
# 2. validation step :
#  no_grad → AUC 계산
# 3. 공통 학습 로직 분리 :
#  train.py 깔끔하게 만들기 위해 필수


import torch
import numpy as np
from sklearn.metrics import roc_auc_score


# =========================================================
# 1 epoch training
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================================================
# validation
# =========================================================
def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # AUC 계산
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    return total_loss / len(loader), auc