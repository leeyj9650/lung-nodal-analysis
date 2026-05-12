from pathlib import Path
import numpy as np
from src.configs.config import RAW_DATA_DIR


# =========================================================
# 2.5D 가능 여부 검사
# =========================================================
def get_index(path):

    return int(path.stem.split("_")[1])


def main():

    root = RAW_DATA_DIR / "slices"
    patients = list(root.glob("LIDC-IDRI-*"))

    total = 0
    possible = 0

    for p in patients:

        files = list(p.glob("*.npy"))
        if len(files) < 3:
            continue

        indices = sorted([get_index(f) for f in files])
        s = set(indices)

        total += 1

        for z in indices:
            if (z - 1 in s) and (z + 1 in s):
                possible += 1
                break

    print("patients:", total)
    print("2.5D possible:", possible)


if __name__ == "__main__":
    main()