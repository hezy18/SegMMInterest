import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader

from PIL import Image
from functools import partial
from transformers import CLIPFeatureExtractor, CLIPVisionModel

import glob
import numpy as np
import os
import random
from tqdm import tqdm
from einops import rearrange

from util_file import LargeHDF5Cache
import traceback

torch.manual_seed(0)

def merge_list_to_tensor(feature_dict, include_keys=None, exclude_keys=None, mode="stack"):
    if include_keys is None:
        include_keys = list(feature_dict.keys())
    if exclude_keys is None:
        exclude_keys = []

    for k in include_keys:
        if k in exclude_keys:
            continue
        if mode == "stack":
            feature_dict[k] = torch.from_numpy(np.stack(feature_dict[k]))
        else:
            feature_dict[k] = torch.from_numpy(np.concatenate(feature_dict[k], axis=0))

    return feature_dict

def collect_features_from_sample_list(sample_list, keys=None):
    if keys is None:
        keys = list(sample_list[0].keys())
    has_single_key = isinstance(keys, str)
    if has_single_key:
        keys = [keys]

    ret_list = []
    for k in keys:
        ret_list += [[s[k] for s in sample_list]]

    if has_single_key:
        return {keys: ret_list[0]}
    else:
        return dict(zip(keys, ret_list))

class HFImageModelWrapper(nn.Module): 
    """enable huggingface image transformer to be used for video inference"""

    def __init__(self, model, extractor, batch_size=16, use_cuda=True) -> None:
        super().__init__()
        self.model = model
        self.extractor = extractor
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    @torch.no_grad()
    def forward(self, raw_data):

        images = [Image.open(path).convert("RGB") for path in raw_data]
        dataloader = DataLoader(images,
                                batch_size=self.batch_size,
                                collate_fn=partial(self.extractor, return_tensors="pt"),
                                num_workers=8,
                                prefetch_factor=6,
                                pin_memory=True)
        outputs_list = []
        for inputs in dataloader:
            if self.use_cuda:
                inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            outputs = self.post_forward(outputs)
            outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            outputs_list += [outputs]

        outputs = merge_list_to_tensor(collect_features_from_sample_list(outputs_list), mode="concat")
        return outputs['last_hidden_state']

    def post_forward(self, outputs):
        return outputs
    

class VisionCLIPWrapper(HFImageModelWrapper):

    def __init__(self,
                 model,
                 extractor,
                 batch_size=16,
                 spatial_reduce=4,
                 temporal_reduce=2,
                 pooling="avg",
                 use_cuda=True) -> None:
        super().__init__(model, extractor, batch_size, use_cuda)
        self.spatial_reduce = spatial_reduce
        self.temporal_reduce = temporal_reduce
        self.pooling = pooling

    def post_forward(self, outputs):
        last_hidden_state = outputs["last_hidden_state"]
        patch_size = int(np.sqrt(last_hidden_state.shape[1]))
        last_hidden_state = rearrange(last_hidden_state[:, 1:, :], "t (h w) d -> d t h w", h=patch_size,
                                      w=patch_size)[None, :]

        pooling = F.avg_pool3d if self.pooling == "avg" else F.max_pool3d
        # spatial_reduce = self.spatial_reduce
        # temporal_reduce = np.minimum(last_hidden_state.shape[2], self.temporal_reduce)
        # last_hidden_state = pooling(last_hidden_state,
        #                             kernel_size=(temporal_reduce, spatial_reduce, spatial_reduce),
        #                             stride=(temporal_reduce, spatial_reduce, spatial_reduce))
        last_hidden_state = pooling(last_hidden_state, kernel_size=(1, last_hidden_state.shape[3], last_hidden_state.shape[4]))

        last_hidden_state = rearrange(last_hidden_state[0], "d t h w -> t h w d")

        last_hidden_state = torch.squeeze(last_hidden_state, dim=-2)  # Remove last dimension
        last_hidden_state = last_hidden_state.reshape(last_hidden_state.shape[0], last_hidden_state.shape[-1])

        # outputs["last_hidden_state"] = last_hidden_state
        # print(outputs)
        return {"last_hidden_state":last_hidden_state}
    

def run_extractor(dataset, hdf5_file):
    try:
        pretrained = 'openai/clip-vit-large-patch14-336'
        model = CLIPVisionModel.from_pretrained(pretrained)
        print(traceback.format_exc())
        extractor = CLIPFeatureExtractor.from_pretrained(pretrained)
    except:
        
        pretrained = "~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
        model = CLIPVisionModel.from_pretrained(pretrained)
        extractor = CLIPFeatureExtractor.from_pretrained(pretrained)
    print('Model Loaded')
    print(model)
    print(extractor)
    
    model_wrapper = VisionCLIPWrapper(model=model, extractor=extractor, batch_size=64, use_cuda=False)
    
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)

    dataloader = tqdm(dataloader)
    
    hdf5_cache = LargeHDF5Cache(hdf5_file, compression="gzip", compression_opts=9)
    
    if os.path.exists(hdf5_file):
        cache_dict = hdf5_cache.load_final()
        print(cache_dict, cache_dict.dtype())
    else:
        cache_dict = {}

    with torch.no_grad():
        for e in dataloader:
            e = e[0]
            video_id = e["video_id"]
            if hdf5_cache.key_exists(video_id):
                print(f"{video_id} exists, skip")
                continue
            if video_id in cache_dict:
                print(cache_dict[video_id].shape)
                continue
            frame_paths = e["frame_paths"]
            print(f"{video_id} inference",len(frame_paths))
            last_hidden_state = model_wrapper(frame_paths)
            # outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            # save_dict = {video_id: outputs}
            save_dict = {video_id: last_hidden_state.cpu().detach().numpy()}
            hdf5_cache.cache_save(save_dict)
            print('cache_saved')
    print('start final save')
    hdf5_cache.final_save()

def load_data(image_root):
    video_paths = glob.glob(os.path.join(image_root, "*"))
    ret_dataset = []
    for video_path in video_paths:
        video_id = os.path.basename(video_path)
        frame_paths = glob.glob(os.path.join(video_path, "*.jpeg"))
        frame_paths = sorted(frame_paths)
        cur_elem = dict(video_id=video_id, frame_paths=frame_paths)
        ret_dataset += [cur_elem]
    return ret_dataset

def load_test_data():
    video_ids = ['1','2','3','4','5','6','7','8','9']
    ret_dataset = []
    for video_id in video_ids:
        frame_paths = ['./data/0.jpeg'] * random.randint(2, 20)
        cur_elem = dict(video_id=video_id, frame_paths=frame_paths)
        ret_dataset += [cur_elem]
    return ret_dataset

def is_default_image(image):
    return np.all(np.array(image) == 0)


if __name__ == '__main__':
    # dataset = load_data(image_root='')
    dataset = load_test_data()
    print(dataset)
    hdf5_file = './data/image_feat.hdf5'
    os.remove(hdf5_file)
    run_extractor(dataset, hdf5_file)