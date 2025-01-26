import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader

from PIL import Image
from functools import partial
from transformers import CLIPFeatureExtractor, CLIPVisionModel
import cv2

import glob
import numpy as np
import os
import random
from tqdm import tqdm
from einops import rearrange

from util_file import LargeHDF5Cache, save_txt
import traceback
import datetime
today=str(datetime.date.today())

Use_Cuda = False
if torch.cuda.is_available():
    Use_Cuda = True

SEED=2024

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Use_Cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic

    

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
    def forward(self, images):
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
        last_hidden_state = pooling(last_hidden_state, kernel_size=(1, last_hidden_state.shape[3], last_hidden_state.shape[4]))

        last_hidden_state = rearrange(last_hidden_state[0], "d t h w -> t h w d")

        last_hidden_state = torch.squeeze(last_hidden_state, dim=-2)  # Remove last dimension
        last_hidden_state = last_hidden_state.reshape(last_hidden_state.shape[0], last_hidden_state.shape[-1])

        # outputs["last_hidden_state"] = last_hidden_state
        # print(outputs)
        return {"last_hidden_state":last_hidden_state}

def is_default_image(image):
    return np.all(np.array(image) == 0)

def run_extractor(dataset, hdf5_file):
    pretrained = 'clip-vit-large-patch14-336'
    model = CLIPVisionModel.from_pretrained(pretrained)
    extractor = CLIPFeatureExtractor.from_pretrained(pretrained)
    print('Model Loaded')
    print(model)
    print(extractor)
    if Use_Cuda:
        model = model.cuda()
    model_wrapper = VisionCLIPWrapper(model=model, extractor=extractor, batch_size=64, use_cuda=Use_Cuda)
    
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)

    dataloader = tqdm(dataloader)
    
    
    hdf5_cache = LargeHDF5Cache(hdf5_file, compression="gzip", compression_opts=9)
    
    cache_dict_prev={}
    for file in ['0_', '1_', '2_', '3_', '4_', '5_7']:
        prev_file='data/SegMM_feat/'+file
        if os.path.exists(prev_file):
            hdf5_cache_prev = LargeHDF5Cache(prev_file, compression="gzip", compression_opts=9)
            cache_dict_prev_tmp= hdf5_cache_prev.load_final()
            cache_dict_prev.update(cache_dict_prev_tmp)

    zero_frame_list = []
    with torch.no_grad():
        # tmp_dict = {}
        # count = 0
        for e in dataloader:
            e = e[0]
            video_id = e["video_id"]
            if hdf5_cache.key_exists(video_id):
                print(f"{video_id} exists, skip")
                continue
            if video_id in cache_dict_prev:
                print(video_id, 'exists', cache_dict_prev[video_id].shape)
                continue
            frame_paths = e["frame_paths"]
            print(f"{video_id} inference",len(frame_paths))
            # images = [Image.open(path).convert("RGB") for path in frame_paths]
            images = []
            for path in frame_paths:
                img = Image.open(path).convert("RGB")
                # if np.all(np.array(img)==0):
                # if not img.getbbox(): # 去除defaul image
                #     print('all zero',path)
                #     zero_frame_list.append(path)
                #     continue_flag = True
                # else:
                images.append(img)
            if len(images)==0:
                zero_frame_list.append('0 frame video: '+str(video_id))
                continue
            last_hidden_state = model_wrapper(images)
            # outputs = {k: v.cpu().detach().numpy() for k, v in outputs.items()}
            # save_dict = {video_id: outputs}
            save_dict = {video_id: last_hidden_state.cpu().detach().numpy()}
            # tmp_dict.update(save_dict)
            # count+=1

            # if count==100:
            hdf5_cache.cache_save(save_dict)
            print('items cache_saved')
            # tmp_dict.clear()
            # count=0
        # if count>0:
        #     hdf5_cache.cache_save(tmp_dict)
        #     print(f'{count} items cache_saved')
    save_txt(zero_frame_list,f'zero_frame_{today}.txt')
    print('start final save')
    hdf5_cache.final_save(time = str(today))

def calculate_frame_ids(time):
    """根据time计算帧的ID列表"""
    frame_ids = [int(i/1000) for i in range(0, int(time), 5000)] 
    return frame_ids

def load_data(pid_list):
    feats_pids = []
    ret_dataset = []
    for pid, duration in pid_list:
        frame_dir_path = f'data/SegMM_frames_per5sec/{pid}'
        if not os.path.exists(frame_dir_path):
            continue
        frame_ids = calculate_frame_ids(duration) # duration_ms
        frame_paths = glob.glob(os.path.join(frame_dir_path, "*.jpg"))
        if len(frame_paths)!=len(frame_ids):
            continue
        frame_paths = sorted(frame_paths)
        cur_elem = dict(video_id=str(pid), frame_paths=frame_paths)
        ret_dataset += [cur_elem]
        feats_pids.append(pid)
    with open('feats_done_pid_list.txt', 'w', encoding='utf-8') as f:
        for pid in feats_pids:
            f.write(str(pid) + '\n')
    print(ret_dataset)
    return ret_dataset


if __name__ == '__main__':
    done_pid_list=[]
    with open('Done_pid_duration_list.txt', 'r', encoding='utf-8') as file:
        for line in file:
            pid, duration = line.strip().split()
            pid, duration = int(pid), int(duration)
            done_pid_list.append((pid, duration))

    group_size = len(done_pid_list) // 8
    if len(done_pid_list) % 8 != 0:
        group_size += 1

    groups = [done_pid_list[i * group_size:(i + 1) * group_size] for i in range(8)]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group_id', type=int, help='0-7')
    parser.add_argument('--feat_name', type=str, help='4-10')

    args, _ = parser.parse_known_args()

    # import pdb; pdb.set_trace()

    group_id = args.group_id # 7
    pid_list = groups[group_id]
    dataset = load_data(pid_list)

    init_seed(SEED)

    hdf5_file = f'data/SegMM_feat/{args.feat_name}' # 10
    run_extractor(dataset, hdf5_file)


