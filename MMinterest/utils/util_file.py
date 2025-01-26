import json

# import joblib
import numpy as np
import pickle
import csv
import os
import subprocess
from typing import Sequence, Mapping
import glob
from tqdm import tqdm
from torch.utils.data import DataLoader
import warnings

import h5py


def load_json(fn):
    with open(fn, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, fn):
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def save_txt(obj: dict, fn):
    with open(fn, 'a', encoding='utf-8') as file:
        file.write('\n'.join(obj))

def load_txt(fn):
    with open(fn, 'r') as file:
        return file.read().splitlines()

# def load_pickle(fn):
#     # return joblib.load(fn)
#     with open(fn, "rb") as f:
#         obj = dill.load(f)
#     return obj


# def save_pickle(obj, fn):
#     # return joblib.dump(obj, fn, protocol=pickle.HIGHEST_PROTOCOL)
#     with open(fn, "wb") as f:
#         dill.dump(obj, f, protocol=dill.HIGHEST_PROTOCOL)


# def run_or_load(func, file_path, cache=True, overwrite=False, args=dict()):
#     # def run_or_loader_wrapper(**kwargs):
#     #     if os.path.exists()
#     #     return fn(**kwargs)
#     # if os.path.exists(file_path)
#     if overwrite or not os.path.exists(file_path):
#         obj = func(**args)
#         if cache:
#             save_pickle(obj, file_path)
#     else:
#         obj = load_pickle(obj)

#     return obj


def load_csv(fn, delimiter=",", has_header=True):
    fr = open(fn, "r")
    read_csv = csv.reader(fr, delimiter=delimiter)

    ret_list = []

    for idx, x in enumerate(read_csv):
        if has_header and idx == 0:
            header = x
            continue
        if has_header:
            ret_list += [{k: v for k, v in zip(header, x)}]
        else:
            ret_list += [x]

    return ret_list


def save_hdf5(obj, fn, **kwargs):
    if isinstance(fn, str):
        with h5py.File(fn, "a") as f:
            save_hdf5_recursive(obj, f, **kwargs)
    elif isinstance(fn, h5py.File):
        save_hdf5_recursive(obj, fn, **kwargs)
    else:
        raise NotImplementedError(f"{type(obj)} to hdf5 not implemented")


def save_hdf5_recursive(kv, cur_handler, **kwargs):
    """convenient saving hierarchical data recursively to hdf5"""
    if isinstance(kv, Sequence):
        kv = {str(idx): v for idx, v in enumerate(kv)}
    for k, v in kv.items():
        if k in cur_handler:
            warnings.warn(f"{k} already exists in {cur_handler}")
        else:
            if isinstance(v, np.ndarray):
                cur_handler.create_dataset(k, data=v, **kwargs)
            elif isinstance(v, Mapping):
                next_handler = cur_handler.create_group(k)
                save_hdf5_recursive(v, next_handler)


def load_hdf5(fn):
    if isinstance(fn, str):
        with h5py.File(fn, "r") as f:
            return load_hdf5_recursive(f)
    elif isinstance(fn, h5py.Group) or isinstance(fn, h5py.Dataset):
        return load_hdf5_recursive(fn)
    else:
        raise NotImplementedError(f"{type(fn)} from hdf5 not implemented")


def load_hdf5_recursive(cur_handler):
    """convenient saving hierarchical data recursively to hdf5"""
    if isinstance(cur_handler, h5py.Group):
        ret_dict = dict()
        for k in cur_handler.keys():
            ret_dict[k] = load_hdf5_recursive(cur_handler[k])
        return ret_dict
    elif isinstance(cur_handler, h5py.Dataset):
        return np.array(cur_handler)
    else:
        raise NotImplementedError(f"{type(cur_handler)} from hdf5 not implemented")

def collate_fn(x):
            return load_hdf5(x[0])

class LargeHDF5Cache:
    """to deal with IO comflict during large scale prediction,
    we mock the behavior of single hdf5 and build hierarchical cache according to hash_key
    this class is only for saving large hdf5 with dense IO
    """

    def __init__(self, hdf5_path, **kwargs):
        self.hdf5_path = hdf5_path
        self.kwargs = kwargs
        self.tmp_dir = hdf5_path + ".tmp"
        self.file_index = 0

        os.makedirs(os.path.dirname(self.hdf5_path), exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

    def key_exists(self, hash_id):
        finish_flag = os.path.join(self.tmp_dir, hash_id + ".hdf5.finish")
        return os.path.exists(finish_flag)

    def cache_save(self, save_dict):
        """save_dict should be like {hash_id: ...}"""
        hash_id = list(save_dict.keys())[0]
        tmp_file = os.path.join(self.tmp_dir, hash_id + ".hdf5")
        subprocess.run(f"rm -rf {tmp_file}", shell=True)  # in case of corrupted tmp file without .finish flag
        save_hdf5(save_dict, tmp_file, **self.kwargs)
        subprocess.run(f"touch {tmp_file}.finish", shell=True)
        
    # def _get_new_file_path(self):
    #     """Generate a new file path for splitting large files."""
    #     self.file_index += 1
    #     return f"{self.hdf5_path.rsplit('.', 1)[0]}_{self.file_index}.hdf5"


    def final_save(self):
        tmp_files = glob.glob(os.path.join(self.tmp_dir, "*.hdf5"))
        result_handle = h5py.File(self.hdf5_path, "a")
        loader = DataLoader(tmp_files,
                            batch_size=1,
                            collate_fn=collate_fn,
                            num_workers=8,
                            prefetch_factor=6)
        for ret_dict in tqdm(loader, desc=f"merging to {self.hdf5_path}"):
            save_hdf5(ret_dict, result_handle)
        result_handle.close()
        subprocess.run(f"rm -rf {self.tmp_dir}", shell=True)
    
    def final_save_multi(self):
        tmp_files = glob.glob(os.path.join(self.tmp_dir, "*.hdf5"))
        current_file_path = self._get_new_file_path()
        result_handle = h5py.File(current_file_path, "a")
        current_file_size = 0

        loader = DataLoader(tmp_files,
                            batch_size=1,
                            collate_fn=collate_fn,
                            num_workers=8,
                            prefetch_factor=6)
        
        for ret_dict in tqdm(loader, desc=f"Merging to {self.hdf5_path}"):
            save_hdf5(ret_dict, result_handle)
            current_file_size = os.path.getsize(current_file_path) / (1024 ** 3)  # Convert to GB

            if current_file_size >= self.max_file_size_gb:
                result_handle.close()
                current_file_path = self._get_new_file_path()
                result_handle = h5py.File(current_file_path, "a")

        result_handle.close()
        subprocess.run(f"rm -rf {self.tmp_dir}", shell=True)

    
    def load_final(self):
        """Load the final merged HDF5 file into memory."""
        return load_hdf5(self.hdf5_path)