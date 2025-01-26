from .logger import get_logger
from omegaconf import OmegaConf
import copy
import hydra

# class Registry:
#     def __init__(self):
#         self._cls_dict = dict()

#     def register(self, name=None):
#         def decorator(cls):
#             # if name is None:
#             #     name = cls.__name__
#             self._cls_dict[name] = cls
#             return cls

#         return decorator

#     def build(self, name, **kwargs):
#         return self._cls_dict[name](**kwargs)

#     def build_cls(self, name):
#         return self._cls_dict[name]

log = get_logger(__name__)

class Registry:
    """mini version of https://github.com/facebookresearch/mmf/blob/main/mmf/common/registry.py"""

    mapping = dict()
    mapping["object"] = dict()

    @classmethod
    def register_cls(cls, name, domain):
        if domain not in cls.mapping:
            cls.mapping[domain] = dict()

        def wrapper(inp_cls):
            if name in cls.mapping[domain]:
                if inp_cls.__name__ != cls.mapping[domain][name].__name__:
                    raise Exception(f"conflict at [{domain}]{name}")
            else:
                """only log at first import"""
                log.info(f"[{domain}] {inp_cls.__name__} registered as {name}")
            cls.mapping[domain][name] = inp_cls
            return inp_cls

        return wrapper

    @classmethod
    def build(cls, _name, _domain, **kwargs):
        assert _name in cls.mapping[_domain], f"no {_name} found in [{_domain}]"
        return cls.mapping[_domain][_name](**kwargs)

    @classmethod
    def build_from_cfg(cls, _cfg, domain):
        cfg = copy.deepcopy(_cfg)
        if hasattr(cfg, "_target_"):
            return hydra.utils.instantiate(cfg)
        elif hasattr(cfg, "type"):
            OmegaConf.resolve(cfg)
            cfg_dict = OmegaConf.to_container(cfg)
            type = cfg_dict.pop("type")  # type: ignore
            return cls.build(type, domain, **cfg_dict)  # type: ignore
        else:
            raise Exception(f"instantiation for config below is unimplemented\n{cfg}")

    @classmethod
    def register_optimizer(cls, _name):
        return cls.register_cls(_name, "optimizer")

    @classmethod
    def build_optimizer(cls, _name, **kwargs):
        return cls.build(_name, "optimizer", **kwargs)

    @classmethod
    def register_scheduler(cls, _name):
        return cls.register_cls(_name, "scheduler")

    @classmethod
    def build_scheduler(cls, _name, **kwargs):
        return cls.build(_name, "scheduler", **kwargs)

    @classmethod
    def register_model(cls, _name):
        return cls.register_cls(_name, "model")

    @classmethod
    def build_model(cls, _name, **kwargs):
        return cls.build(_name, "model", **kwargs)

    @classmethod
    def register_datapipe(cls, _name):
        return cls.register_cls(_name, "datapipe")

    @classmethod
    def build_datapipe(cls, _name, **kwargs):
        return cls.build(_name, "datapipe", **kwargs)

    @classmethod
    def register_collater(cls, _name):
        return cls.register_cls(_name, "collater")

    @classmethod
    def build_collater(cls, _name, **kwargs):
        return cls.build(_name, "collater", **kwargs)

    @classmethod
    def register_processor(cls, _name):
        return cls.register_cls(_name, "processor")

    @classmethod
    def build_processor(cls, _name, **kwargs):
        return cls.build(_name, "processor", **kwargs)

    @classmethod
    def register_task(cls, _name):
        return cls.register_cls(_name, "task")

    @classmethod
    def build_task(cls, _name, **kwargs):
        return cls.build(_name, "task", **kwargs)

    @classmethod
    def register_metric(cls, _name):
        return cls.register_cls(_name, "metric")

    @classmethod
    def build_metric(cls, _name, **kwargs):
        return cls.build(_name, "metric", **kwargs)

    @classmethod
    def register_loss(cls, _name):
        return cls.register_cls(_name, "loss")

    @classmethod
    def build_loss(cls, _name, **kwargs):
        return cls.build(_name, "loss", **kwargs)

    @classmethod
    def register_pipeline(cls, _name):
        return cls.register_cls(_name, "pipeline")

    @classmethod
    def build_pipeline(cls, _name, **kwargs):
        return cls.build(_name, "pipeline", **kwargs)
    
    @classmethod
    def register_function(cls, _name):
        return cls.register_cls(_name, "function")

    @classmethod
    def call_function(cls, _name, **kwargs):
        return cls.build(_name, "function", **kwargs)

    """
    @classmethod
    def register_(cls, _name):
        return cls.register_cls(_name, "")

    @classmethod
    def build_(cls, _name, **kwargs):
        return cls.build(_name, "", **kwargs)
    """

    @classmethod
    def register_object(cls, name, obj, verbose=False):
        if name in cls.mapping["object"]:
            if id(obj) != id(cls.mapping["object"][name]):
                raise Exception(f"conflict at [object]{name}")
        if verbose:
            log.info(f"[object] {id(obj)} registered as {name}")

        cls.mapping["object"][name] = obj

    def set_object(cls, name, obj):
        cls.mapping["object"][name] = obj

    @classmethod
    def get_object(cls, name, default_val=None):
        # assert name in cls.mapping["object"], f"no {name} found in [object]"
        return cls.mapping["object"].get(name, default_val)

    @classmethod
    def destry_object(cls, name, verbose=False):
        if verbose:
            log.info(f"[object] {id(cls.mapping['object'][name])} with name {name} destroyed")
        del cls.mapping["object"][name]

registry = Registry()
global_upload = registry.register_object
global_get = registry.get_object
global_set = registry.set_object
