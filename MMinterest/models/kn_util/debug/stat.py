import numpy as np
from ..general import registry

class Statistics:
    vals = dict()

    @classmethod
    def update(cls, name, val):
        if name not in cls.vals:
            cls.vals[name] = []
        cls.vals[name] += [val]

    @classmethod
    def compute(cls, name, op, delete=False):
        ret_val = None

        if op == "max":
            ret_val = np.max(cls.vals[name])
        elif op == "avg":
            ret_val = np.mean(cls.vals[name])
        elif op == "min":
            ret_val = np.min(cls.vals[name])
        
        # primitive type to make omegaconf happy
        if type(ret_val).__name__.startswith("int"):
            ret_val = int(ret_val)  # type: ignore
        elif type(ret_val).__name__.startswith("float"):
            ret_val = float(ret_val)  # type: ignore
        
        assert ret_val is not None, f"op {op} not supported"
        
        if delete:
            del cls.vals[name]
        
        return ret_val

stat = Statistics()