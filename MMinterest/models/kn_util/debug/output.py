from typing import Sequence, Mapping
import torch
import copy
import numpy as np
from ..basic.file import save_pickle
import json


def recursive_lambda(x, _lambda):
    if isinstance(x, Sequence):
        ret_list = []
        for v in x:
            ret_list += [recursive_lambda(v, _lambda)]
        return ret_list
    elif isinstance(x, Mapping):
        ret_dict = dict()
        for k, v in x.items():
            ret_dict[k] = recursive_lambda(v, _lambda)
        return ret_dict
    else:
        return _lambda(x)


def list_gradient_norm(mod):
    param_dict = dict(mod.named_parameters())

    def get_gradient_norm(param):
        return param.grad.norm()

    grad_norm_dict = recursive_lambda(param_dict, _lambda=get_gradient_norm)
    print(json.dumps(grad_norm_dict, indent=4, sort_keys=True))


def explore_content(x, name="default", depth=0, max_depth=3, str_len_limit=20, list_limit=10, print_str=True):
    ret_str = ""
    sep = "    "
    if isinstance(x, str):
        if len(x) < str_len_limit:
            ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{x})\n"
        else:
            ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"

    elif isinstance(x, Sequence):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
        if not isinstance(x, str):
            for idx, v in enumerate(x):
                ret_str += explore_content(v, name=str(idx), depth=depth + 1, max_depth=max_depth)  # type: ignore
                if idx > list_limit:
                    ret_str += sep * (depth + 1) + "...\n"
                    break

    elif isinstance(x, Mapping):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}{sep}{len(x)} elements)\n"
        for k, v in x.items():
            ret_str += explore_content(v, name=k, depth=depth + 1, max_depth=max_depth)

    elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__}[{x.dtype}]{sep}{tuple(x.shape)})" + "\n"
    else:
        ret_str += sep * depth + f"{name}{sep}({type(x).__name__})\n"
        if hasattr(x, "__dict__") and depth < max_depth:  # in case it's not a class
            # ret_str += "\t" * depth + f"{name}({type(x).__name__}"
            for k, v in vars(x).items():
                ret_str += explore_content(v, k, depth=depth + 1, max_depth=max_depth)

    if depth != 0:
        return ret_str
    else:
        if print_str:
            print(ret_str)
        else:
            return ret_str


# def dict_diff(before, now):
#     add = dict()
#     delete = dict()
#     all_keys = set(before.keys()) | set(now.keys())

#     for k in all_keys:
#         if k in now and k in before:
#             if id(now[k]) != id(before[k]):
#                 add[k] = now[k]
#                 delete[k] = before[k]
#         elif k not in before and k in now:
#             add[k] = now[k]
#         elif k in before and k not in now:
#             delete[k] = before[k]
#         else:
#             raise Exception()

#     delete_str = explore_content(delete, "DELETED", max_depth=0)
#     add_str = explore_content(add, "ADDED", max_depth=0)

#     del add
#     del delete
#     del all_keys

#     return "-" * 60 + "\n" + delete_str + "+" * 60 + "\n" + add_str
