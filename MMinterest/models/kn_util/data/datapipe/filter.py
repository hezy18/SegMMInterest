from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe
import h5py

@functional_datapipe("filter_by_hdf5_key")
class HDF5KeyFilter(IterDataPipe):
    def __init__(self, src_pipeline, hdf5_file, key_template="{}") -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.hdf5_file = hdf5_file
        self.key_template = key_template
    
    def __iter__(self):
        hdf5_handler = h5py.File(self.hdf5_file, "r")
        for x in self.src_pipeline:
            cur_key = self.key_template.format(**x)
            if cur_key in hdf5_handler:
                yield x
        hdf5_handler.close()