import torch
def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
