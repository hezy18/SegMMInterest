from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
from kn_util.data import general_pad

@functional_datapipe("pad_sequence")
class SequencePadder(IterDataPipe):

    def __init__(self, src_pipeline, from_key=None, return_mask=True, **pad_args) -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.pad_args = pad_args
        self.return_mask = return_mask
        self.from_key = from_key

    def __iter__(self):
        for x in self.src_pipeline:
            data = x[self.from_key]
            if self.return_mask:
                padded_data, mask = general_pad(data, return_mask=self.return_mask, **self.pad_args)
                # yield padded_data, mask
                x.update({self.from_key + ".pad": padded_data, self.from_key + ".mask": mask})
            else:
                padded_data = general_pad(data, return_mask=self.return_mask, **self.pad_args)
                # yield padded_data
                x.update({self.from_key + ".pad": padded_data})
            yield x
