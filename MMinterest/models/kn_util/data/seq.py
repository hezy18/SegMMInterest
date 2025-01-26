import numpy as np


def generate_sample_indices(tot_len, max_len=None, stride=None):
    assert (max_len is not None) ^ (stride is not None)
    if max_len is not None:
        stride = int(np.ceil((tot_len - 1) / (max_len - 1)))

    indices = list(range(0, tot_len - 1, stride)) + [tot_len - 1]
    return indices


def slice_by_axis(data, _slice, axis):
    num_axes = len(data.shape)
    slices = tuple([_slice if _ == axis else slice(0, data.shape[_]) for _ in range(num_axes)])
    return data[slices]


def reduce_segment(data, st_idx, ed_idx, axis=0, mode="avgpool"):
    span = ed_idx - st_idx
    if st_idx == ed_idx:
        cur_frames = slice_by_axis(data, slice(st_idx, st_idx+1), axis=axis)
        return cur_frames
    cur_frames = slice_by_axis(data, slice(st_idx, ed_idx), axis=axis)
    if mode == "maxpool":
        sampled_frame = np.max(cur_frames, axis=axis, keepdims=True)
    elif mode == "avgpool":
        sampled_frame = np.mean(cur_frames, axis=axis, keepdims=True)
    elif mode == "center":
        center_idx = span // 2
        sampled_frame = slice_by_axis(cur_frames, slice(center_idx, center_idx + 1), axis=axis)
    elif mode == "random":
        random_idx = np.random.choice(np.arange(ed_idx - st_idx))
        sampled_frame = slice_by_axis(cur_frames, slice(random_idx, random_idx + 1), axis=axis)
    elif mode == "tail":
        tail_idx = ed_idx - 1
        sampled_frame = slice_by_axis(cur_frames, slice(tail_idx, tail_idx + 1), axis=axis)
    return sampled_frame


def general_sample_sequence(data, axis=0, stride="constant", max_len=None, mode="avgpool"):
    """
    suppose data.shape[axis] = tot_len and we want to devide them into N segments
    in first case, we keep the stride as a constant
    a) mode = maxpool / avgpool
       for each segment, we pooling all features into a single feature (using max or avg)
    b) mode = sample
       for each segment, we keep both the head and tail frames, it can be seen as choose the tail frame for each segment and add head frame
    c) mode = center / random
       for each segment, we select one random frame or its center frame
    in second case, we uniformly split sequence and round splitting points to nearest indices
    a) 
    """

    tot_len = data.shape[axis]
    ret_frames = []
    # stride = "constant"
    if stride == "constant" or isinstance(stride, int):
        # length cannot be fixed if stride is a constant
        if stride == "constant":
            assert max_len is not None
            stride = np.ceil(tot_len / max_len)
        idxs = np.arange(0, tot_len, stride)
        import ipdb; ipdb.set_trace() #FIXME
    # stride = "round"
    # length will be fixed to max_len
    else:
        assert max_len
        idxs = np.arange(0, max_len + 1, 1.0) / max_len * tot_len
        idxs = np.round(idxs).astype(np.int32)
        idxs[idxs >= tot_len] = tot_len - 1
        for i in range(len(idxs) - 1):
            st_idx = idxs[i]
            ed_idx = idxs[i+1]
            sampled_frame = reduce_segment(data, st_idx, ed_idx, axis=axis, mode=mode)
            ret_frames.append(sampled_frame)
        ret_frames = np.concatenate(ret_frames, axis=axis)
        return ret_frames
