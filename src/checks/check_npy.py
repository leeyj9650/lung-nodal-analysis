import numpy as np
from pathlib import Path
import pandas as pd

from src.configs.config import RAW_DATA_DIR

# =========================================================
# npy 상태 확인
# =========================================================
def main():

    npy_files = list((RAW_DATA_DIR / "slices").rglob("*.npy"))

    sample = np.load(npy_files[0])

    print("===== NPY CHECK =====")
    print("shape:", sample.shape)
    print("dtype:", sample.dtype)
    print("min:", sample.min())
    print("max:", sample.max())
    print("mean:", sample.mean())


if __name__ == "__main__":
    main()