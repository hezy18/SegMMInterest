import re
from torch import nn
from functools import wraps, partial

from einops import rearrange, reduce, repeat, einsum

# adapted from https://github.com/lucidrains/einops-exts

# checking shape
# @nils-werner
# https://github.com/arogozhnikov/einops/issues/168#issuecomment-1042933838

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)

# do same einops operations on a list of tensors

def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)
repeat_many = _many(repeat)
reduce_many = _many(reduce)
