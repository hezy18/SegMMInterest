from .lazy import LazyCall as L

import copy
from omegaconf import DictConfig


def serializable(_config, depth=0):
    if depth == 0:
        config = copy.deepcopy(_config)
    else:
        config = _config
    if not isinstance(config, DictConfig):
        return
    else:
        for k in config.keys():
            if isinstance(config[k], type):
                config[k] = str(config)
                return

            serializable(config[k], depth + 1)
    if depth == 0:
        return config


def eval_str_impl(s):
    return eval(s)


def eval_str(s):
    return L(eval_str_impl)(s=s)


from hydra.utils import instantiate as _instantiate
import copy


def instantiate(_cfg, _convert_="none", **kwargs):
    cfg = copy.deepcopy(_cfg)
    cfg.update(kwargs)
    return _instantiate(cfg, _convert_=_convert_)
