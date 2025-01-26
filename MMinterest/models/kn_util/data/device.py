from typing import Mapping, Sequence
import torch 
def collection_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, str):
        return batch
    elif isinstance(batch, Mapping):
        return {k: collection_to_device(v, device) for k,v in batch.items()}
    elif isinstance(batch, Sequence):
        return [collection_to_device(x, device) for x in batch]
    else:
        return batch

def collection_apply(batch, fn=lambda x: x):
    if isinstance(batch, list):
        for i in range(len(batch)):
            batch[i] = collection_apply(batch[i], fn)
    if isinstance(batch, dict):
        for k in batch.keys():
            batch[k] = collection_apply(batch[k], fn)
    try:
        return fn(batch)
    except:
        return batch