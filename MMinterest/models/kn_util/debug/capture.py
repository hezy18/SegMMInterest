from ..basic.file import save_pickle
from ..basic import get_logger, global_upload
from .output import explore_content
from functools import partial


def capture_output(_lambda, to_global=False):
    def out_wrapper(fn):
        def wrapper(*args, **kwargs):
            ret = fn(*args, **kwargs)
            lambda_dict = _lambda(ret)
            if to_global:
                for k, v in lambda_dict.items():
                    global_upload(k, v)
            return ret
        
        return wrapper
    return out_wrapper

def capture_forward_and_print(name=None, return_names=None, mode="explore", **kwargs):
    if mode == "explore":
        if return_names is None:
            _lambda = partial(explore_content, name=name, print_str=True, **kwargs)
        else:
            def to_dict(x):
                x_dict = dict()
                for k, v in zip(return_names, x):
                    x_dict[k] = v
                return x_dict
            _lambda = lambda x: partial(explore_content, name=name, print_str=True, **kwargs)(to_dict(x))
    elif mode == "print":
        _lambda = print
    return capture_output(_lambda)