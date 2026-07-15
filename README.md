## 실행 순서

### 전처리
parse lidc annotations.py   ← XML 파싱 (XML annotation을 읽어서 결절 정보를 만든다.)
    └ 실행방법 : python -m src.preprocessing.parse_lidc_annotations
    └ 실행시 data/processed/nudule_info.json

match dicom.py              ← DICOM 매칭 (DICOM 폴더와 매칭해서 CT 메타정보를 추가)
    └ 실행방법 : python -m src.preprocessing.match_dicom
    └ 실행시 data/processed/nudule_info_clean.json   (DICOM 경로와 CT 정보가 정상적으로 붙은 환자/결절 정보)
             outputs/figures/dicom_histogram.png     (rows, cols, slice 수, spacing 분포 확인용 그림)

export nifti.py             ← Nifti 변환 + Segmentation mask 생성 + HU
    └ 실행방법 : python -m src.preprocessing.export_nifti
    └ 실행시 data/processed/nifti
            data/processed/coord_violations.json   
            data/processed/seg_empty_subjects.json 


### 학습
src/preprocessing/labels.py                  ← 라벨생성
    └ 실행방법 : python -m src.preprocessing.labels
    └ 실행시 data/processed/labels.csv                                     

src/preprocessing/split.py                   ← 환자별 분리
    └ 실행방법 : python -m src.preprocessing.split
    └ 실행시 data/processed/split.json 


src/preprocessing.make_npy.py   ※ npy 새로 만들시 rm -rf data/processed/npy_cache   !!!
    └ 실행방법(fixed) : python -m src.preprocessing.make_npy --crop_mode fixed --crop_size 64 --image_size 128

    └ 실행방법(dynamic) : python -m src.preprocessing.make_npy --dynamic_crop --image_size 128 --crop_scale 4 --crop_min_mm 32 --crop_max_mm 96
       
    └ 실행시
    data/processed/npy_cache/images/*.npy (실제 학습 입력 이미지)
    data/processed/npy_cache/samples.json (각 npy 파일 정보)
    data/processed/npy_cache/config.json  (npy 생성 당시 설정 기록)

rm -rf outputs/predictions/*
rm -rf outputs/logs/*

#동적 crop
python -m src.preprocessing.make_npy --dynamic_crop --crop_size 64 --image_size 128
python -m src.training.train_lidc_2d_npy_weight  --dynamic_crop --crop_size 64 --image_size 128

python -m src.preprocessing.make_npy --dynamic_crop --crop_size 64 --image_size 64
python -m src.training.train_lidc_2d_npy_weight  --dynamic_crop --crop_size 64 --image_size 64


#정적 crop 
python -m src.preprocessing.make_npy --crop_size 64 --image_size 128
python -m src.training.train_lidc_2d_npy_weight --crop_size 64 --image_size 128

python -m src.preprocessing.make_npy --crop_size 64 --image_size 64
python -m src.training.train_lidc_2d_npy_weight --crop_size 64 --image_size 64
