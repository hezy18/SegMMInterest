import torch
import torch.nn as nn
import torch.nn.functional as F


def copy_weight(from_model, to_model, weight_map):
    from_weight = {name: param for name, param in from_model.named_parameters()}
    for name, param in to_model:
        for to_prefix, from_prefix in weight_map.items():
            if name.startswith(to_prefix):
                from_name = from_prefix + name[len(to_prefix) :]
                assert from_name in from_weight
