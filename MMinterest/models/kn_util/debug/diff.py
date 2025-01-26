import torch
import numpy as np
import os
import dill
from ..data import collection_apply
import copy
from typing import Mapping, Sequence
from termcolor import colored
import os
import os.path as osp
from ..basic import load_pickle, save_pickle


def diff(a, b, index=""):
    if type(a) != type(b):
        print(colored(index, "blue") + colored(":type_diff:", "red") + f"a({type(a)} b({type(b)}))")
        return
    elif isinstance(a, np.ndarray):
        if a.shape != b.shape:
            print(colored(index, "blue") + colored(":np_shape_diff:", "red") + f"a({a.shape}) b({b.shape})")
            return
        if not np.allclose(a, b):
            print(colored(index, "blue") + colored(":np_val_diff:", "red") + f"err_norm={np.linalg.norm(a-b)}")
    elif isinstance(a, torch.Tensor):
        if a.shape != b.shape:
            print(colored(index, "blue") + colored(":torch_shape_diff:", "red") + f"a({a.shape}) b({b.shape})")
            return
        if not torch.allclose(a, b):
            print(colored(index, "blue") + colored(":torch_val_diff:", "red") + f"err_norm={(a-b).norm()}")
    elif isinstance(a, Mapping):
        if a.keys() != b.keys():
            print(colored(index, "blue") + colored(":key_diff:", "red") + f"a({a.keys()}) b({b.keys()})")
            return
        for k in a.keys():
            diff(a[k], b[k], index=index + f".{k}")
        return
    elif isinstance(a, (str, int, float)):
        if a != b:
            print(colored(index, "blue") + colored(f":{type(a)}_diff:", "red") + f"a({a}) b({b})")
    elif isinstance(a, Sequence):
        if len(a) != len(b):
            print(colored(index, "blue") + colored(":size_diff:", "red") + f"a({len(a)} b({len(b)}))")
            return
        for idx in range(len(a)):
            diff(a[idx], b[idx], index=index + f".{idx}")
    else:
        try:
            if a != b:
                print(colored(index, "blue") + colored(":unknown_diff:", "red") + f"a({type(a)}) b({type(b)})")
        except:
            pass

def get_variable(name, debug_dir=osp.expanduser("~/_DEBUG")):
    fn = osp.join(debug_dir, name + ".pkl")
    loaded = load_pickle(fn)
    return loaded

def sync_diff(obj, name, debug_dir=osp.expanduser("~/_DEBUG")):
    # for debugging randomness
    # sync and compare between two processes
    os.makedirs(debug_dir, exist_ok=True)
    fn = osp.join(debug_dir, name + ".pkl")
    if os.getenv("KN_DEBUG_SEND", False):
        save_pickle(obj, fn)
        print(f"{fn} saved")
    if os.getenv("KN_DEBUG_RECV", False):
        loaded = load_pickle(fn)
        diff(obj, loaded, index=name)