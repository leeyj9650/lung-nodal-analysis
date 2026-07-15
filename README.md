# 폐결절 주변 맥락 인지형 악성도 이진 분류 모델 (Rebuild)

> [cite_start]**"단순 결절 정보를 넘어 흉막, 혈관 등 주변 해부학적 맥락(Context)을 인지하여 6mm 미만 소결절의 판독 정확도를 극대화한 초경량 2.5D 딥러닝 솔루션"** 
>
> [cite_start]본 프로젝트는 LIDC-IDRI 표준 CT 영상 데이터셋을 활용하여, 형태적 정보가 부족한 '소결절(Small Nodule)'의 한계를 극복하기 위해 결절 내부뿐만 아니라 주변 맥락 정보를 함께 학습하는 **'2.5D 입력 기반 주변 맥락 인지형 딥러닝 모델'** 개발을 목적으로 합니다.

---

## 📅 프로젝트 개요
- [cite_start]**개발 기간**: 2026.05.13 ~ 2026.06.19 (약 5주) [cite: 4, 5]
- [cite_start]**개발 인원**: 6인 (팀 프로젝트) 
- [cite_start]**과정명**: 융합 메디컬 AI with 스마트 웰니스 [cite: 3]
- [cite_start]**팀명**: 리빌드 (Rebuild) 

---

## 🛠️ 기술 스택 및 아키텍처

### 1. Technology Stack
- [cite_start]**Environment**: Visual Studio Code, Git / GitHub 
- [cite_start]**Libraries**: PyTorch, pydicom, nibabel, simpleITK, scikit-learn, pandas, numpy, matplotlib 
- [cite_start]**Dataset**: LIDC-IDRI 표준 데이터셋 (CT Modality) 

### 2. 핵심 아키텍처 (2.5D 입력 및 주변 맥락 인지)
- [cite_start]**2.5D Multi-Slice Patch Extraction**: 메모리 효율성을 극대화한 2.5D 패치 추출 기법 적용 
- [cite_start]**Model**: Gated-Dilated Network (GDN) 기반 아키텍처에 CBAM(Convolutional Block Attention Module) 및 pos_weight 최적화 적용 
- [cite_start]**Ablation Models**: ConvNeXt, Dual-ConvNeXt 비교 실험 

---

## 🎯 주요 구현 기능 (Core Features)

1. [cite_start]**전문의 합의 기반 데이터 자동 정제**: 다수의 의학 전문의 주석(XML annotation)을 파싱하여 노이즈 없는 고품질 데이터 자동 필터링 
2. [cite_start]**2.5D 패치 추출 & 정규화**: [-1000, 400] HU(Hounsfield Unit) 클리핑 및 등방성 1mm 리샘플링을 통한 2.5D 슬라이스 데이터셋 구축 
3. [cite_start]**주변 맥락 인지형 AI 판독**: 결절 주변부(흉막, 혈관 연결성 등)의 해부학적 상관관계를 분석하는 딥러닝 분류기 
4. [cite_start]**결절 크기별 성능 평가 시스템**: 소/중/대분류 세부 그룹(Subgroup Analysis)별 AUC-ROC, 민감도(Sensitivity), 특이도(Specificity) 추적 
5. [cite_start]**Grad-CAM 시각화**: AI가 의사 결정 과정에서 결절 주변의 어떤 맥락적 특징을 보고 악성으로 진단했는지 설명 가능성(XAI) 제공 

---

## 👤 나의 기여 및 담당 역할 (My Role)

### 1. Ablation Study를 통한 모델 최적화 및 비교 검증 (Dual-ConvNeXt & ConvNeXt)
- [cite_start]프로젝트 가설인 *"주변 맥락 정보가 악성도 판단에 미치는 영향"*을 증명하기 위해 **ConvNeXt 및 Dual-ConvNeXt 백본 기반의 비교 실험(Ablation Study)을 주도적으로 설계하고 실행**했습니다.
- [cite_start]단 10만 개의 파라미터를 사용하는 가볍고 강건한 GDN 모델이 무거운 대형 모델인 Dual-ConvNeXt의 한계(과적합)를 극복하고 더 우수한 성능을 도출할 수 있도록 **비교군 모델들의 성능 지표를 엄격하게 추적**했습니다.

### 2. 프로젝트 리소스 관리 및 형상 관리 주도 (자료 백업 & 수행일지)
- [cite_start]의료 및 딥러닝 프로젝트의 특성상 수많은 실험 가중치(`.pth`)와 대용량 파일 변경이 발생하는 환경에서, **안정적인 깃허브 브랜치 전략 및 데이터 백업 관리 파이프라인을 담당**했습니다.
- [cite_start]개발 진행 과정 및 실험 조건 변경 이력을 체계적으로 기록하는 **수행일지를 철저히 작성**하여 팀원 간의 유기적인 협업과 실험 재현성을 극대화했습니다.

---

## ⚙️ 실행 순서 (Execution Guide)

### 1. 데이터 전처리 (Preprocessing)

#### ① LIDC XML Annotation 파싱
- 전문의의 XML annotation을 해석하여 결절 메타데이터 정보를 자동 생성합니다.

```
python -m src.preprocessing.parse_lidc_annotations
# 출력 결과물: data/processed/nodule_info.json
```

#### ② DICOM 매칭 및 통계 시각화
- 파싱된 정보와 원본 DICOM 폴더를 매칭하여 CT 메타데이터를 추가하고, 데이터 세부 스페이싱 분포를 확인합니다.
```
python -m src.preprocessing.match_dicom
# 출력 결과물: 
#  - data/processed/nodule_info_clean.json
#  - outputs/figures/dicom_histogram.png (데이터 분포 확인용)
```

#### ③ NIfTI 변환 및 3D 세그멘테이션 마스크 생성
- CT 영상의 HU 정규화 및 리샘플링을 거쳐 NIfTI 포맷 변환 및 마스크 파일을 추출합니다.
```
python -m src.preprocessing.export_nifti
# 출력 결과물: 
#  - data/processed/nifti
#  - data/processed/coord_violations.json
#  - data/processed/seg_empty_subjects.json
```

### 2. 데이터 분류 및 패치 학습 준비

#### ① 데이터 라벨 생성
```
python -m src.preprocessing.labels
# 출력 결과물: data/processed/labels.csv
```

#### ② 환자별 분리 (Train/Val/Test = 70/15/15%)
- 데이터 누수를 방지하기 위해 환자 단위(Patient-level) 분할을 보장합니다.
```
python -m src.preprocessing.split
# 출력 결과물: data/processed/split.json
```

#### ③ 2.5D npy 패치 데이터셋 빌드 (npy_cache)
- 주의: npy를 새로 빌드할 경우 기존 캐시 디렉토리를 완전히 비워주세요: rm -rf data/processed/npy_cache
```
- 정적 크롭 모드 (Fixed Crop)
python -m src.preprocessing.make_npy --crop_mode fixed --crop_size 64 --image_size 128

- 동적 크롭 모드 (Dynamic Crop)
python -m src.preprocessing.make_npy --dynamic_crop --image_size 128 --crop_scale 4 --crop_min_mm 32 --crop_max_mm 96

- 출력 결과물:
data/processed/npy_cache/images/*.npy (실제 입력 이미지)
data/processed/npy_cache/samples.json (메타 정보)
data/processed/npy_cache/config.json (생성 시점 설정값)
```

#### 3. 학습 및 최적화 비교 실험 (Training)
- 학습을 새로 시작하기 전 예측 결과물과 로그 폴더를 정리해 줍니다.
```
rm -rf outputs/predictions/*
rm -rf outputs/logs/*

- 옵션 A. 동적 크롭 (Dynamic Crop) 실험 파이프라인

# Image Size 128 실험
python -m src.preprocessing.make_npy --dynamic_crop --crop_size 64 --image_size 128
python -m src.training.train_lidc_2d_npy_weight --dynamic_crop --crop_size 64 --image_size 128

# Image Size 64 실험
python -m src.preprocessing.make_npy --dynamic_crop --crop_size 64 --image_size 64
python -m src.training.train_lidc_2d_npy_weight --dynamic_crop --crop_size 64 --image_size 64

- 옵션 B. 정적 크롭 (Fixed Crop) 실험 파이프라인

# Image Size 128 실험
python -m src.preprocessing.make_npy --crop_size 64 --image_size 128
python -m src.training.train_lidc_2d_npy_weight --crop_size 64 --image_size 128

# Image Size 64 실험
python -m src.preprocessing.make_npy --crop_size 64 --image_size 64
python -m src.training.train_lidc_2d_npy_weight --crop_size 64 --image_size 64

```

### 프로젝트 주요 성과 및 기대 효과 (Expected Results)
- 초경량·고성능 모델 설계: 파라미터 수가 단 10만 개에 불과한 초경량 아키텍처로 무거운 Dual-ConvNeXt 모델 수준을 능가하는 Test AUC 약 0.89 및 민감도 약 0.84 달성.  
- 주변 맥락(Context)의 임상적 가치 입증: 판별이 가장 어려웠던 6mm 미만의 소결절 진단 성능(Small AUC)을 기존 0.75에서 0.84 수준까지 대폭 향상시켜 학술적 가설 증명.  
- 의료 AI 민주화 가능성 확보: 초경량 구조 설계 덕분에 값비싼 엔터프라이즈급 GPU 장비가 없는 1, 2차 소규모 의료기관의 엣지 기기에서도 즉각적이고 정확한 폐결절 스크리닝 구동 가능.  
- Grad-CAM 기반 신뢰도 구축: 결절 단면 분석을 넘어 혈관 연결성 등 주변 해부학적 맥락을 반영했음을 증명하는 설명 가능 대시보드 리포트 확보.
