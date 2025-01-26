import torch


def detach_collections(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().detach()
    if isinstance(x, dict):
        for k in x:
            x[k] = detach_collections(x[k])
    if isinstance(x, list):
        for k in x:
            k = detach_collections(k)
    return x
