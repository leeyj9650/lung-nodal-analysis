# LIDC-IDRI Lung Nodule Classification Project

## 1. 프로젝트 개요

LIDC-IDRI 폐 CT 데이터를 이용해 결절의 악성 여부를 분류하는 프로젝트입니다.

분류 클래스:

- 0: benign
- 1: malignant

현재 사용 데이터:

- 이미 생성된 `.npy` CT slice
---

## 2. 프로젝트 구조
```
Project2/
│
├── data/                 ← 원본/전처리 데이터 저장
│   ├── raw/              ← 원본 LIDC 데이터
│   │   └── lidc-idri/
│   │
│   └── processed/        ← 학습 가능한 형태로 가공된 데이터
│       ├── method_a/
│       │   ├── labels.csv
│       │   └── npy/
│       │
│       └── method_c/
│           ├── labels.csv
│           └── npy/
│
├── outputs/              ← 실험 결과 저장
│   ├── checkpoints/      ← 학습된 모델(.pth)
│   ├── logs/             ← json 로드, loss, auc 기록
│   ├── figures/          ← ROC Curve, Confusion matrix
│   ├── gradcam/          ← Grad-Cam 이미지
│   └── predictions/      ← 예측 결과 csv
│
├── src/                  ← 실제 코드
│   ├── configs/          ← 설정 관리
│   │   └── config.py
│   │
│   ├── datasets/         ← PyTorch Dataset/Dataloader
│   │   └── dataset.py    
│   │
│   ├── models/           ← 모델 정의
│   │   └── resnet.py
│   │
│   ├── preprocessing/          ← 전처리 코드(데이터셋 생성 담당)
│   │   ├── preprocess.py       ← dataset 생성 (labels.csv생성, npy 복사)
│   │   └── label_policy.py     ← 라벨 기준 분리 
│   │
│   ├── training/         ← 학습 관련
│   │   ├── train.py
│   │   └── engine.py     ← 공통 train/validate 함수
│   │
│   ├── evaluation/       ← 평가 코드
│   │   ├── evaluate.py
│   │   └── metrics.py    ← 계산 함수 모음 (AUC, F1, Sensitivity, Specificity)
│   │
│   ├── explainability/   ← 설명 가능성(XAI)
│   │   └── gradcam.py
│   │
│   ├── utils/            ← 공통 유틸 함수
│   │   ├── seed.py       ← random seed 고정
│   │   └── device.py     ← cuda/cpu 설정
│   │
│   └── checks/           ← 데이터 검사 코드
│       ├── check_npy.py  ← npy 상태 확인
│       └── check_25d_possible.py   ← 2.5D 가능한지 검사
│
├── requirements.txt
├── README.md
└── .gitignore
```
## 3.실행순서

conda activate lidc_resnet

cd ~/projects/Project2

# preprocess
python -m src.preprocessing.preprocess

# train
python -m src.training.train --method a
python -m src.training.train --method c

# evaluation
python -m src.evaluation.evaluate --method a
python -m src.evaluation.evaluate --method c
.
# gradcam
python -m src.explainability.gradcam --method a --num_images 5
python -m src.explainability.gradcam --method c --num_images 5

## 4. 변수 설명
1. val loss : 검증 데이터에서의 error
     └ 모델 generaliztion 성능 확인

    핵심 포인트
    | train loss ↓, val loss ↓  | 정상 |
    | train lostt↓, val laoss↑  | 과적합 |
    | train lostt↑, val laoss↑  | 학습 부족 |

2. val auc : “모델이 benign vs malignant를 얼마나 잘 구분하는지”
    AUC = Area Under ROC Curve

    해석 기준
    AUC	    상태
    0.5	    랜덤
    0.6	    약함
    0.7	    보통
    0.8+	좋음

3. CONFUSION MATRIX (CM)

    구조
	Pred 0	Pred 1
    True 0	TN	FP
    True 1	FN	TP

    TN(True Positive)   : 정상을 정상이라고 맞춤
    TP (True Positive)  : 암을 암이라고 맞춤
    FP (False Positive) : 정상인데 암이라고 오판
    FN (False Negative) : 암인데 정상이라고 판단

4. sensitivity (민감도) : “암 환자를 암이라고 맞춘 비율”

5. specificity (특이도) : “정상 사람을 정상이라고 맞춘 비율”

6. F1 score : precision(정밀도) + recall(재현율) 균형
    사용 이유 : 데이터 imbalance 있을 때 중요

    상황	                        의미
    Precision ↑ / Recall ↓	    보수적인 모델 (잘 안 찍지만 맞추는 것만 찍음)
    Precision ↓ / Recall ↑	    공격적인 모델 (많이 잡지만 오탐 많음)
    둘 다 균형	                 F1 ↑ (가장 이상적)

7. ROC Curve : threshold 바꿔가면서 성능 변화

    구성
    X축: FPR(False Positive Rate) = 실제 음성인데 양성으로 잘못 판단한 비율
    Y축: TPRTPR (True Positive Rate) = 실제 양성을 맞춘 비율 (Recall과 동일 개념)

    그래프 해석
    곡선	    의미
    좌상단      붙음 좋음(TPR 높고 FPR 낮음)
    대각선	    랜덤
