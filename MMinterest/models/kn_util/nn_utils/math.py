import numpy as np
import torch

def gaussian(mean, sigma, n_position):
    """
    discrete gaussian over `n_position` indices
    return [n_position, ]
    """

    sigma = sigma
    mean = mean

    idx_tensor = np.arange(n_position)
    # (bsz, n_position)
    gauss = np.exp(-np.square(idx_tensor - mean) / (2 * np.square(sigma)))

    return gauss

def gaussian_torch(mean, sigma, n_position):
    # mean, sigma can be arbitrary shape
    # return (..., n_position)
    assert mean.shape == sigma.shape
    to_shape = mean.shape
    mean_flatten = mean.flatten().unsqueeze(-1)
    sigma_flatten = sigma.flatten().unsqueeze(-1)
    idx_tensor = torch.arange(n_position, device=mean.device)[None, :]
    gauss = torch.exp(-torch.square(idx_tensor - mean_flatten) / (2 * torch.square(sigma_flatten)))
    gauss = gauss.reshape((*to_shape, -1))
    return gauss


def inverse_sigmoid_torch(x: torch.Tensor, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)

