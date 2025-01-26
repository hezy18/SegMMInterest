from ..lazy import LazyCall as L
from torch.optim import AdamW

def adamw(lr=3e-4, betas=(0.9, 0.999), weight_decay=0.0001):
    return L(AdamW)(params=None, lr=lr, betas=betas, weight_decay=0.0001)
