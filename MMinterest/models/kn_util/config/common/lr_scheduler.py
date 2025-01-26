from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from kn_util.config import LazyCall as L


def reduce_lr_on_plateau(mode, factor=0.9, patience=20, verbose=True, min_lr=1e-6):
    return L(ReduceLROnPlateau)(optimizer=None,
                                factor=factor,
                                mode=mode,
                                patience=patience,
                                verbose=verbose,
                                min_lr=min_lr)


# def cosine_annealing_lr(eta_min=1e-2, verbose=True):
#     return CosineAnnealingLR(eta_min=eta_min, verbose=verbose)
