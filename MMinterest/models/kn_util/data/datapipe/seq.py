from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import functional_datapipe
from kn_util.data import general_sample_sequence

@functional_datapipe("sample_sequence")
class SequenceSampler(IterDataPipe):
    def __init__(self, src_pipeline, from_key, inplace=True, **sample_args) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.sample_args = sample_args
        self.from_key = from_key
        self.inplace = inplace

    def __iter__(self):
        for x in self.src_pipeline:
            seq = x[self.from_key]
            sampled_seq = general_sample_sequence(seq, **self.sample_args)
            if self.inplace:
                x[self.from_key] = sampled_seq
            else:
                x[self.from_key + ".sample"] = sampled_seq
            yield x

