import torch
import torch.nn as nn
from torchvision import models


# =========================================================
# ResNet50 Model 생성 함수
# =========================================================
def create_model(num_classes=2, pretrained=True):

    # -----------------------------------------------------
    # ImageNet pretrained weight 사용 여부
    # -----------------------------------------------------
    if pretrained:
        weights = models.ResNet50_Weights.DEFAULT
    else:
        weights = None

    # -----------------------------------------------------
    # ResNet50 로드
    # -----------------------------------------------------
    model = models.resnet50(weights=weights)

    # -----------------------------------------------------
    # 마지막 Fully Connected Layer 변경
    # -----------------------------------------------------
    # 기존: 1000 class (ImageNet)
    # 변경: 2 class (benign / malignant)
    # -----------------------------------------------------
    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model


# =========================================================
# 실행 테스트 → python -m src.models.resnet
# =========================================================
if __name__ == "__main__":

    model = create_model(pretrained=True)

    dummy = torch.randn(4, 3, 224, 224)

    output = model(dummy)

    print("output shape:", output.shape)