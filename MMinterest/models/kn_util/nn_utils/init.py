import torch
import torch.nn as nn
import math
from functools import partial


def filter_params(m, name_filter_fn=lambda n: True, param_filter_fn=lambda p: True, requires_grad=True):
    return [
        p for n, p in m.named_parameters()
        if name_filter_fn(n) and param_filter_fn(p) and p.requires_grad == requires_grad
    ]


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


def init_weight(m, method="kaiming"):
    _uniform_dict = {
        "kaiming": partial(nn.init.kaiming_uniform_, nonlinearity="relu"),
        "xavier": nn.init.xavier_uniform_,
    }
    uniform_ = _uniform_dict[method]
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        if method == "xavier":
            uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif method == "kaiming":
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def init_module(module):
    if hasattr(module, "init_weight"):
        return
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # module.weight.data.normal_(mean=0.0, std=cfg["initializer_range"])
        nn.init.xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def init_children(m):
    for children in m.children():
        init_module(children)
