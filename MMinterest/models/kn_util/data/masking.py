import numpy as np

def mask_safe(mask):
    tot_len = mask.shape[0]
    mask_cnt = np.sum(mask.astype(np.int32))
    range_i = np.arange(tot_len)
    
    if tot_len == mask_cnt or mask_cnt == 0:
        idx = np.random.choice(range_i)
        mask[idx] = 1 - mask[idx]
    
    return mask