# src/utils/data_augment.py

"""
적용 효과
[기존 기하학적 증강]
1. 좌우 뒤집기 (Horizontal Flip)
2. 랜덤 각도 회전 (Random Rotation)
3. 미세한 평행이동 및 크기 조절 (Random Affine - Translation & Scale)

[신규 의료용 질감/음영 증강 - 과적합 방지 최전선]
4. 가우시안 블러 (Gaussian Blur) - 경계선 외우기 방지
5. 명암비 및 밝기 조절 (Color Jitter - Brightness & Contrast) - 다양한 장비 대응
6. 랜덤 해상도 저하 (Random Downsampling & Upsampling) - 픽셀 외우기 방지
7. 가우시안 노이즈 주입 (Gaussian Noise) - 형태적 과적합 방어선

"""

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

class Medical25DAugment:
    def __init__(self, p: float = 0.5):
        """
        의료 2.5D 데이터셋을 위한 안전한 데이터 증강 클래스
        Args:
            p (float): 증강 기술이 전체적으로 적용될 확률 (기본값 50%)
        """
        self.p = p
        
        # [기존 기하학적 변환]
        self.rotate = T.RandomRotation(degrees=(-15, 15))
        self.affine = T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05))
        
        # [신규 질감 변환 수치 세팅]
        # 커널 크기는 3x3 고정, 시그마(흐림 정도)는 0.1~1.0 사이로 미세하게 설정하여 암의 형태를 파괴하지 않습니다.
        self.blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image (torch.Tensor): [3, H, W] 형태의 2.5D 이미지 텐서 (0.0 ~ 1.0 사이로 정규화된 상태 가정)
        Returns:
            torch.Tensor: 증강이 적용된 이미지 텐서
        """
        # 전체 확률에 따라 증강을 적용할지 말지 결정합니다.
        if random.random() > self.p:
            return image

        # ------------------------------------------------------------
        # 🌟 STEP 1. 기하학적 증강 (위치 및 방향 흔들기)
        # ------------------------------------------------------------
        # 1. 좌우 뒤집기
        if random.random() > 0.5:
            image = F.hflip(image)

        # 2. 미세한 회전 및 평행이동 적용 (3개 슬라이스 일괄 적용)
        image = self.rotate(image)
        image = self.affine(image)

        # ------------------------------------------------------------
        # 🌟 STEP 2. 의료용 질감 및 음영 증강 (과적합 철벽 방어)
        # ------------------------------------------------------------
        # 무작위로 한 가지만 선택하여 적용함으로써, 이미지가 과도하게 뭉개지는 것을 방지합니다.
        cfg_choice = random.choice(['blur', 'jitter', 'downsample', 'none'])
        
        original_shape = (image.shape[1], image.shape[2]) # 원래 이미지의 (H, W) 저장

        if cfg_choice == 'blur':
            # 3. 가우시안 블러: 경계선을 희미하게 만들어 테두리 암기를 방지합니다.
            image = self.blur(image)
            
        elif cfg_choice == 'jitter':
            # 4. 명암비 및 밝기 조절: CT 장비별 밝기 차이를 시뮬레이션합니다.
            # 흑백(Grayscale) 성분이므로 brightness와 contrast만 미세하게(±10%) 조절합니다.
            brightness_factor = random.uniform(0.9, 1.1)
            contrast_factor = random.uniform(0.9, 1.1)
            image = F.adjust_brightness(image, brightness_factor)
            image = F.adjust_contrast(image, contrast_factor)
            
        elif cfg_choice == 'downsample':
            # 5. 랜덤 해상도 저하: 화질을 뭉개서 픽셀값 자체를 외우는 꼼수를 차단합니다.
            # 임의로 크기를 반토막(64->32 등) 냈다가 보간법(Bilinear)을 통해 원래 크기로 되돌립니다.
            low_h, low_w = int(original_shape[0] * 0.5), int(original_shape[1] * 0.5)
            image = F.resize(image, [low_h, low_w], interpolation=T.InterpolationMode.BILINEAR)
            image = F.resize(image, list(original_shape), interpolation=T.InterpolationMode.BILINEAR)

        # ------------------------------------------------------------
        # 🌟 STEP 3. 미세 가우시안 노이즈 주입 (마지막 필터)
        # ------------------------------------------------------------
        # 6. 모델이 0.99로 정답을 외워버리는 현상을 막는 가장 강력한 브레이크입니다.
        if random.random() > 0.5:
            # std=0.02는 아주 미세한 지직거림입니다. 모델이 돋보기를 들고 픽셀을 외우지 못하게 만듭니다.
            noise = torch.randn_like(image) * 0.02
            image = image + noise
            # 노이즈가 더해져 0.0 미만이나 1.0 초과로 튀는 픽셀값을 안전하게 제한(Clip)해 줍니다.
            image = torch.clamp(image, 0.0, 1.0)

        return image