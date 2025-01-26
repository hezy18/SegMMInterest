from torchdata.datapipes.iter import IterDataPipe
from torch.utils.data import functional_datapipe
import numpy as np
import h5py


@functional_datapipe("load_hdf5")
class HDF5Loader(IterDataPipe):

    def __init__(self, src_pipeline, hdf5_file, key_template="{}", output_key_prefix="") -> None:
        super().__init__()
        self.src_pipeline = src_pipeline
        self.hdf5_file = hdf5_file
        self.key_template = key_template
        self.output_key_prefix = output_key_prefix

    def __iter__(self):
        hdf5_handler = h5py.File(self.hdf5_file, "r")
        for x in self.src_pipeline:
            cur_key = self.key_template.format(**x)
            if isinstance(hdf5_handler[cur_key], h5py.Group):
                cur_group = hdf5_handler[cur_key]
                for k in cur_group:
                    x[self.output_key_prefix + f".{k}.hdf5"] = np.array(cur_group[k])
            else:
                x[self.output_key_prefix + ".hdf5"] = np.array(hdf5_handler[cur_key])
            yield x
        hdf5_handler.close()