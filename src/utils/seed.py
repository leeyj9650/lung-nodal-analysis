import torch
import random
import numpy as np

# =========================================================
# 재현성 고정
# =========================================================
def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)