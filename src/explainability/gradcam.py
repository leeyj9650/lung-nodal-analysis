# =========================================================
# Grad-CAM Visualization
# =========================================================
# 역할:
# - 학습된 모델이 CT 이미지의 어느 부분을 중요하게 봤는지 시각화
# - heatmap 생성
# - overlay 이미지 생성
#
# 실행:
# python -m src.explainability.gradcam --method a --num_images 5
#
# 결과 저장:
# outputs/gradcam/
#
# 개선 사항:
# ✔ timestamp 추가
# ✔ 실행별 누적 저장
# ✔ original / heatmap / overlay 저장
# ✔ prediction probability 표시
# ✔ 파일명 충돌 방지
# =========================================================

import argparse
from datetime import datetime

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.configs.config import *
from src.datasets.dataset import get_dataloaders
from src.models.resnet import create_model


# =========================================================
# Device 설정
# =========================================================
def get_device():

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# =========================================================
# Grad-CAM 클래스
# =========================================================
class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        # gradient 저장
        self.gradients = None

        # activation map 저장
        self.activations = None

        # =================================================
        # Hook 등록
        # =================================================

        # forward 시 activation 저장
        self.target_layer.register_forward_hook(
            self.forward_hook
        )

        # backward 시 gradient 저장
        self.target_layer.register_full_backward_hook(
            self.backward_hook
        )

    # =====================================================
    # Activation 저장
    # =====================================================
    def forward_hook(self, module, input, output):

        self.activations = output

    # =====================================================
    # Gradient 저장
    # =====================================================
    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    # =====================================================
    # CAM 생성
    # =====================================================
    def generate(self, x, class_idx=None):

        # gradient 초기화
        self.model.zero_grad()

        # forward
        output = self.model(x)

        # prediction class 자동 선택
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # target score
        score = output[:, class_idx]

        # backward
        score.backward()

        # gradient
        gradients = self.gradients

        # activation map
        activations = self.activations

        # =================================================
        # channel-wise importance
        # =================================================
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # weighted sum
        cam = (weights * activations).sum(dim=1)

        # negative 제거
        cam = torch.relu(cam)

        # tensor -> numpy
        cam = cam.squeeze().detach().cpu().numpy()

        # resize
        cam = cv2.resize(
            cam,
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        # normalize
        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min() + 1e-8
        )

        return cam, class_idx


# =========================================================
# Grad-CAM 저장
# =========================================================
def save_gradcam(
    image,
    cam,
    label,
    pred,
    prob,
    idx,
    method,
    timestamp
):

    # =====================================================
    # tensor -> numpy
    # =====================================================
    image = image.squeeze().cpu().numpy()

    # [3,H,W] -> [H,W,3]
    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = np.clip(image, 0, 1)

    # =====================================================
    # heatmap 생성
    # =====================================================
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    heatmap = heatmap / 255.0

    # =====================================================
    # overlay
    # =====================================================
    overlay = (
        0.5 * image
        + 0.5 * heatmap
    )

    overlay = np.clip(overlay, 0, 1)

    # =====================================================
    # figure 생성
    # =====================================================
    plt.figure(figsize=(12, 4))

    # =====================================================
    # Original
    # =====================================================
    plt.subplot(1, 3, 1)

    plt.imshow(image)

    plt.title("Original")

    plt.axis("off")

    # =====================================================
    # Heatmap
    # =====================================================
    plt.subplot(1, 3, 2)

    plt.imshow(cam, cmap="jet")

    plt.title("Grad-CAM")

    plt.axis("off")

    # =====================================================
    # Overlay
    # =====================================================
    plt.subplot(1, 3, 3)

    plt.imshow(overlay)

    plt.title(
        f"L:{label} / P:{pred}\nProb:{prob:.4f}"
    )

    plt.axis("off")

    # =====================================================
    # 파일명
    # =====================================================
    save_path = (
        GRADCAM_DIR
        / f"{timestamp}_method_{method}_{idx}.png"
    )

    # =====================================================
    # 저장
    # =====================================================
    plt.savefig(
        save_path,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[Saved] {save_path}")


# =========================================================
# Main
# =========================================================
def main(method="a", num_images=5):

    # =====================================================
    # 실행 시간
    # =====================================================
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    # =====================================================
    # Device
    # =====================================================
    device = get_device()

    print("===================================")
    print("Grad-CAM Start")
    print("===================================")

    print("Device:", device)
    print("Method:", method)

    # =====================================================
    # Test Loader
    # =====================================================
    _, _, test_loader = get_dataloaders(method)

    # =====================================================
    # Model
    # =====================================================
    model = create_model(
        pretrained=False
    )

    # =====================================================
    # Weight Load
    # =====================================================
    model.load_state_dict(
        torch.load(
            CHECKPOINT_DIR
            / f"best_model_{method}.pth",
            map_location=device
        )
    )

    model = model.to(device)

    model.eval()

    # =====================================================
    # ResNet 마지막 conv layer
    # =====================================================
    target_layer = model.layer4[-1]

    # =====================================================
    # GradCAM 객체
    # =====================================================
    gradcam = GradCAM(
        model,
        target_layer
    )

    count = 0

    # =====================================================
    # inference
    # =====================================================
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        for i in range(images.size(0)):

            # single image
            x = images[i:i+1]

            # =================================================
            # prediction
            # =================================================
            with torch.no_grad():

                output = model(x)

                probs = torch.softmax(
                    output,
                    dim=1
                )

                pred = output.argmax(
                    dim=1
                ).item()

                prob = probs[0, pred].item()

            # =================================================
            # CAM 생성
            # =================================================
            cam, _ = gradcam.generate(
                x,
                pred
            )

            # =================================================
            # 저장
            # =================================================
            save_gradcam(
                image=x,
                cam=cam,
                label=labels[i].item(),
                pred=pred,
                prob=prob,
                idx=count,
                method=method,
                timestamp=timestamp
            )

            count += 1

            print(
                f"[{count}/{num_images}] done"
            )

            # =================================================
            # 종료 조건
            # =================================================
            if count >= num_images:

                print()
                print("Grad-CAM 완료")

                return


# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # method
    parser.add_argument(
        "--method",
        type=str,
        default="a",
        choices=["a", "c"]
    )

    # 저장 이미지 개수
    parser.add_argument(
        "--num_images",
        type=int,
        default=5
    )

    args = parser.parse_args()

    main(
        method=args.method,
        num_images=args.num_images
    )