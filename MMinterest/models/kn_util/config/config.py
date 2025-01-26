from omegaconf import OmegaConf
import os.path as osp


def load_config(cfg_path):
    cfg = OmegaConf.load(cfg_path)
    base_cfgs = []
    if hasattr(cfg, "_base_"):
        for base_cfg_path in cfg._base_:
            base_cfg_path = osp.join(osp.abspath(osp.dirname(cfg_path)), base_cfg_path)
            base_cfgs += [load_config(base_cfg_path)]

        base_cfgs += [cfg]
        cfg = OmegaConf.unsafe_merge(*base_cfgs)

    return cfg
