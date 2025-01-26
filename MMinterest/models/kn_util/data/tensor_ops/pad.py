from kn_util.data.seq import slice_by_axis
import numpy as np
import torch


def _general_pad_arr_pt(arr, axis, to_length,fill_value=None, return_mask=False):
    to_shape = list(arr.shape)
    to_shape[axis] = to_length
    shape_dim = len(to_shape)
    meta = dict(device=arr.device,
                dtype=arr.dtype)
    
    # create full array
    if fill_value == "last":
        # fill with last value
        cur_fill_value = slice_by_axis(arr, _slice=slice(-1, None), axis=axis)
        tile_args = tuple([1 if _ != axis else to_length for _ in range(shape_dim)])
        full_arr = cur_fill_value.repeat(*tile_args)
    else:
        full_arr = torch.full(to_shape, fill_value=fill_value, **meta)

    sub_slices = tuple([slice(0, arr.shape[_]) for _ in range(shape_dim)])
    full_arr[sub_slices] = arr
    ret_arr = full_arr

    if return_mask:
        full_arr = torch.zeros(to_shape, dtype=torch.bool, device=arr.device)
        full_arr[sub_slices] = True
        flatten_slices = tuple([slice(0, to_length) if _ == axis else 0 for _ in range(shape_dim)])
        ret_mask = full_arr[flatten_slices]
    
    return ret_arr, ret_mask if return_mask else ret_arr


def _general_pad_arr_np(arr, axis, to_length,fill_value=None, return_mask=False):
    to_shape = list(arr.shape)
    to_shape[axis] = to_length
    shape_dim = len(to_shape)
    arr_dtype = arr.dtype
    
    # create full array
    if fill_value == "last":
        # fill with last value
        cur_fill_value = slice_by_axis(arr, _slice=slice(-1, None), axis=axis)
        tile_args = tuple([1 if _ != axis else to_length for _ in range(shape_dim)])
        full_arr = np.tile(cur_fill_value, tile_args)
    else:
        full_arr = np.full(to_shape, fill_value=fill_value, dtype=arr_dtype)

    sub_slices = tuple([slice(0, arr.shape[_]) for _ in range(shape_dim)])
    full_arr[sub_slices] = arr
    ret_arr = full_arr

    if return_mask:
        full_arr = np.zeros(to_shape, dtype=bool)
        full_arr[sub_slices] = True
        flatten_slices = tuple([slice(0, to_length) if _ == axis else 0 for _ in range(shape_dim)])
        ret_mask += [full_arr[flatten_slices]]
    
    return ret_arr, ret_mask if return_mask else ret_arr

def general_pad_arr(arr, *args, **kwargs):
    backend =  "pt" if isinstance(arr, torch.Tensor) else "np"
    if backend == "pt":
        return _general_pad_arr_pt(arr, *args, **kwargs)
    elif backend == "np":
        return _general_pad_arr_np(arr,*args, **kwargs)
    else:
        raise ValueError(f"backend {backend} not supported")


