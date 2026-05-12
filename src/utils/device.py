import torch

# =========================================================
# device 자동 선택
# =========================================================
def get_device():

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")