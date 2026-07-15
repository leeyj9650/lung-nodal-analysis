# src/checks/check_split.py
#
# ─── 변경 사항 ───────────────────────────────────────────────────────────────
#   NIfTI 검사: 루트 하드코딩 → labels.csv의 nifti_path 컬럼 사용
#   이유: 다중 시리즈 환자는 series_0/ 서브폴더에 저장되므로
#         루트에서 찾으면 항상 "없음"으로 잘못 나옴
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   python -m src.checks.check_split

import json
from pathlib import Path
import pandas as pd
from src.configs.config import PROCESSED_ROOT


def check_split() -> None:
    labels_csv = PROCESSED_ROOT / 'labels.csv'
    split_json = PROCESSED_ROOT / 'split.json'

    for path in [labels_csv, split_json]:
        if not path.exists():
            raise FileNotFoundError(f'파일 없음: {path}')

    df = pd.read_csv(labels_csv)
    with open(split_json) as f:
        split = json.load(f)

    train_set = set(split['train'])
    val_set   = set(split['val'])
    test_set  = set(split['test'])
    all_split = train_set | val_set | test_set
    all_subj  = set(df['subject_id'].unique())

    print('\n====== Split 무결성 검사 ======')

    # 검사 1: 중복 없음
    tv = train_set & val_set
    tt = train_set & test_set
    vt = val_set   & test_set
    if tv or tt or vt:
        print(f'❌ [FAIL] 중복 subject 존재: train∩val={tv}, train∩test={tt}, val∩test={vt}')
    else:
        print('✅ [PASS] 중복 없음 (data leakage 없음)')

    # 검사 2: 누락 없음
    missing = all_subj - all_split
    extra   = all_split - all_subj
    if missing or extra:
        print(f'❌ [FAIL] subject 불일치 | 누락={missing} | 초과={extra}')
    else:
        print(f'✅ [PASS] 누락 없음 (전체 {len(all_subj)}명 포함)')

    # 검사 3: 레이블 분포
    print('\n── 레이블 분포 (결절 기준) ──')
    for fold_name, ids in [('train', split['train']), ('val', split['val']), ('test', split['test'])]:
        fold_df = df[df['subject_id'].isin(ids)]
        n_ben   = (fold_df['label'] == 0).sum()
        n_mal   = (fold_df['label'] == 1).sum()
        n_total = len(fold_df)
        ratio   = f'{n_ben/n_mal:.2f}:1' if n_mal > 0 else 'N/A'
        print(f'  {fold_name:5s}: {n_total:4d}개 | '
              f'양성:{n_ben}({n_ben/n_total*100:.1f}%) '
              f'악성:{n_mal}({n_mal/n_total*100:.1f}%) | 비율:{ratio}')

    # 검사 4: NIfTI 파일 존재 여부
    # labels.csv의 nifti_path 컬럼을 사용 → 다중 시리즈 환자 정확하게 검사
    print('\n── NIfTI 파일 존재 확인 ──')
    if 'nifti_path' not in df.columns:
        print('⚠ nifti_path 컬럼 없음 → python -m src.preprocessing.labels 재실행 필요')
    else:
        split_df     = df[df['subject_id'].isin(all_split)]
        missing_rows = split_df[~split_df['nifti_path'].apply(lambda p: Path(p).exists())]

        if missing_rows.empty:
            print(f'✅ [PASS] 모든 결절의 nifti_path 존재')
        else:
            missing_subj = sorted(missing_rows['subject_id'].unique())
            print(f'❌ [WARN] nifti_path 없는 결절: {len(missing_rows)}개 ({len(missing_subj)}명)')
            for sid in missing_subj:
                rows = missing_rows[missing_rows['subject_id'] == sid]
                # 해당 환자의 nifti_path 예시 출력
                example_path = rows['nifti_path'].iloc[0]
                print(f'   {sid}: {len(rows)}개 결절 → {example_path}')

    print('\n====== 검사 완료 ======\n')


if __name__ == '__main__':
    check_split()