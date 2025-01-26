from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import copy

@functional_datapipe("collect")
class Collect(IterDataPipe):
    def __init__(self, src_pipeline, from_keys=[], to_keys=None) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.from_keys = from_keys
        if to_keys is not None:
            self.to_keys = to_keys
        else:
            self.to_keys = from_keys
    
    def __iter__(self):
        for x in self.src_pipeline:
            ret_dict = dict()
            for k, to_k in zip(self.from_keys, self.to_keys):
                ret_dict[to_k] = x[k]
            yield ret_dict

@functional_datapipe("rename")
class Rename(IterDataPipe):
    def __init__(self, src_pipeline, from_keys=[], to_keys=None) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.from_keys = from_keys
        self.to_keys = to_keys
    
    def __iter__(self):
        """
        suppose we have A -> B, B -> C
        the result will be {B: A, C: B, A: A}
        the correctness of destination will be guaranteed first
        """
        for x in self.src_pipeline:
            ret_dict = copy.copy(x)
            for k in self.from_keys:
                ret_dict.pop(k)
            for k, to_k in zip(self.from_keys, self.to_keys):
                ret_dict[to_k] = x[k]
            yield ret_dict