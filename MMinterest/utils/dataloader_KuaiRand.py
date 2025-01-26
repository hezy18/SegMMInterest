import subprocess
import re
import os
import numpy as np
from PIL import Image, ImageOps
from typing import Dict, List, Optional
import copy

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader

import traceback
import json

import pandas as pd
import argparse
import time
import datetime
import time
import random

import warnings
warnings.filterwarnings('ignore')


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


class BaseReaderSeq_KuaiRand(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='dataset/KuaiRand/',
                            help='Input data dir.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--data', type=str, default='inter')
        # parser.add_argument('--dict_path', type=str, default='user_input_dict.json')

        parser.add_argument('--history_max', type=int, default=50,
							help='Maximum length of history.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.data = args.data
        # self.dict_path = args.dict_path
        
        # with open(os.path.join(self.prefix, self.dict_path), 'r') as f:
        #     self.user_input_dict = json.load(f)

        self.history_max = args.history_max
        
        if os.path.exists(self.prefix + 'train' +'_his.csv'):
            self.data_df = dict()
            for key in ['train', 'dev', 'test']:
                self.data_df[key] = pd.read_csv(self.prefix + key +'_his.csv', sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time_ms'])

            print(self.data_df[key].columns)
            print('Dataset statistics...')
            key_columns = ['user_id','video_id','time_ms','play_time_ms_x']
            self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
            self.n_users = len(self.all_df['user_id'].unique())  # Count distinct user_ids
            self.n_items = len(self.all_df['video_id'].unique())
            print('"# user": {}, "# item": {}, "# entry": {}'.format(
                self.n_users - 1, self.n_items - 1, len(self.all_df)))
            self.max_users = self.all_df['user_id'].max()
            self.max_items = self.all_df['video_id'].max()
            print('"max user id": {}, "max item id": {}'.format(
                self.max_users, self.max_items))
        else:
            self._read_data()  
            self._append_his_info()

            self._get_history()

            for key in ['train', 'dev', 'test']:
                self.data_df[key] = self.data_df[key].sort_values(by='time_ms')
                self.data_df[key].to_csv(self.prefix + key +'_his.csv', sep=self.sep, index=False)

    def _get_history(self):
        for key in ['train', 'dev', 'test']:
            self.data_df[key]['history_items'] = [None] * len(self.data_df[key])
            self.data_df[key]['history_playing'] = [None] * len(self.data_df[key])
            self.data_df[key]['history_lengths'] = 0
            for index, row in self.data_df[key].iterrows():
                uid = row['user_id']
                pos = row['position']
                user_seq = self.user_his[uid][:pos]
                user_seq = user_seq[-self.history_max:]
                if len(user_seq)>0:
                    self.data_df[key]['history_items'][index]  = np.array([x[0] for x in user_seq])
                    self.data_df[key]['history_playing'][index] = np.array([x[1] for x in user_seq])
                    self.data_df[key]['history_lengths'][index]= len(user_seq)
            print(self.data_df[key].head())

    def _append_his_info(self):
        """
        self.user_his: store user history sequence [(i1,t1), (i1,t2), ...]
        add the 'position' of each interaction in user_his to data_df
        """
        print('Appending history info...')
        sort_df = self.all_df.sort_values(by=['time_ms', 'user_id'], kind='mergesort')
        print(sort_df.head)
        position = list()
        self.user_his = dict()  # store the already seen sequence of each user
        #['user_id','video_id','time_ms','duration_ms','play_time_ms','label_1D']]
        for uid, iid, playing in zip(sort_df['user_id'], sort_df['video_id'], sort_df['play_time_ms']):
            if uid not in self.user_his:
                self.user_his[uid] = list()
            position.append(len(self.user_his[uid]))
            self.user_his[uid].append((iid, playing))
        sort_df['position'] = position
        for key in ['train', 'dev', 'test']:
            self.data_df[key] = pd.merge(
                left=self.data_df[key], right=sort_df, how='left',
                on=['user_id', 'video_id', 'time_ms'])
        del sort_df 

    def _read_data(self):
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            print(self.prefix + key +'.csv')
            self.data_df[key] = pd.read_csv(self.prefix + key +'.csv', sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time_ms'])

        print(self.data_df[key].columns)
        print('Dataset statistics...')
        key_columns = ['user_id','video_id','time_ms','play_time_ms']
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        self.n_users = len(self.all_df['user_id'].unique())  # Count distinct user_ids
        self.n_items = len(self.all_df['video_id'].unique())
        self.max_users = self.all_df['user_id'].max()
        self.max_items = self.all_df['video_id'].max()
        print('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        print('"max user id": {}, "max item id": {}'.format(
            self.max_users, self.max_items))

def custom_resize(image, target_size, padding_color=(255, 255, 255)):
    """
    Resize and pad an image to the target size.
    
    Parameters:
    image (PIL.Image.Image): The input image.
    target_size (tuple): The target size as (width, height).
    padding_color (tuple): The color for padding, default is white.
    
    Returns:
    PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate the scaling factor to maintain aspect ratio
    scale = min(target_width / original_width, target_height / original_height)
    
    # Resize the image with the scaling factor
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Calculate padding to center the resized image
    pad_left = (target_width - new_width) // 2
    pad_top = (target_height - new_height) // 2
    pad_right = target_width - new_width - pad_left
    pad_bottom = target_height - new_height - pad_top

    # Pad the resized image to the target size
    padded_image = ImageOps.expand(resized_image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=padding_color)
    return padded_image


class FrameDatasetSeq_KuaiRand(IterableDataset):
    def __init__(self,corpus:BaseReaderSeq_KuaiRand, 
                shuffle=True, phase='train', # for dataset
                 image_resize=True, target_hw_shape=(224, 224), do_scale_image_to_01=True, 
                 verbose=True):
        super(FrameDatasetSeq_KuaiRand, self).__init__()
        self.do_scale_image_to_01 = do_scale_image_to_01
        self.image_resize = image_resize
        self.target_hw_shape = target_hw_shape
        self.shuffle = shuffle
        self.verbose = verbose
        
        self.photo_max_image = 40
        self.user_max_image = 100
        
        self.df = corpus.data_df[phase]
    
    def _calculate_frame_ids(self,duration_ms):
        frame_ids = [int(i/1000) for i in range(0, int(duration_ms), 5000)]  # 每5000毫秒（5秒）一个帧
        return frame_ids

    def _print(self, *args, **kargs):
        if self.verbose:
            print(*args, **kargs)
            
    def _read_image(self,file_path):
        with Image.open(file_path) as img:
            if self.image_resize: img = custom_resize(img, self.target_hw_shape[::-1])
            return np.array(img)
        
    def _concatenate_images(self,image_list, max_image, axis=0, fill_value=0, ):
        img_shape = image_list[0].shape
        padding_image = np.full(img_shape, fill_value, dtype=np.uint8)
        while len(image_list) < max_image:
            image_list.append(padding_image)
        return np.stack(image_list[:max_image], axis=axis)
    
    def _try_parse_heads(self, head_names):
        self.key2index = {}
        for head_name in head_names:
            head_index = head_names.index(head_name)
            self.key2index[head_name] = head_index
        return self.key2index
    
    def _pad_label_list(self, label_str, max_length, pad_value=-2):
        split_list = label_str.strip('[').strip(']').split(' ')
        label_list = [int(item.strip()) for item in split_list if item.strip()]
        if len(label_list) >= max_length:
            padded_list = label_list[:max_length]
        else:
            padding = [pad_value] * (max_length - len(label_list))
            padded_lst = label_list + padding
            padded_list = padded_lst
        return padded_list
    
    def _getitem(self):
        df = self.df
        
        head_names = df.head(0).columns.to_list()
        # print(head_names)
        key2index = self._try_parse_heads(head_names)
        head_names = ['user_id','video_id','time_ms','label_1D']
        samples_BN = df.to_numpy()
        if self.shuffle: np.random.shuffle(samples_BN)
        
        # self._print(len(samples_BN))
        for sample_i in samples_BN:
            play_time = sample_i[key2index['play_time_ms_x']]
            duration = sample_i[key2index['duration_ms']]
            history_items = sample_i[key2index['history_items']]
            history_playing = sample_i[key2index['history_playing']]
            history_length = sample_i[key2index['history_lengths']]
            frame_ids = self._calculate_frame_ids(duration)
            result_dict = {'play_time':int(play_time/5000),'duration':int(duration/5000)}

            valid_sample = True
            for key in head_names:
                sample_val = sample_i[key2index[key]]
                self._print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),sample_val)
                if 'video' in key: 
                    photo_id = sample_val
                    feature_list = []
                    mask = np.zeros(self.photo_max_image, dtype=bool) 
                    mask[:len(frame_ids)] = True
                    result_dict['photo_mask'] = mask
                    result_dict['photo_id'] = sample_val
                            
                elif 'user' in key:
                    result_dict['user_mask'] =  np.ones(self.user_max_image, dtype=bool) 
                    result_dict['user_id'] = sample_val
                    
                elif 'label' in key:
                    padded_list = self._pad_label_list(sample_val, max_length=self.photo_max_image)
                    result_dict['label'] = np.array(padded_list, dtype=np.int)  # ,dtype=torch.int
                elif 'time_ms' in key:
                    result_dict['time_ms'] = sample_val
            if valid_sample:
                self._print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),result_dict)
                yield result_dict


    def __iter__(self) -> Dict[str, torch.Tensor]:
        for item in self._getitem():
            yield item


class DataCollator(object):
    def __call__(self, batch):
        assert len(batch)
        # print('DataCollator',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        keys = list(batch[0].keys())
        collated_batch = {}
        for k in keys:
            tic = time.perf_counter()
            collated_batch[k] = torch.from_numpy(np.stack([item[k] for item in batch]))
            toc = time.perf_counter()
            print(f"{k} shape is {collated_batch[k].shape}, cost {toc - tic} second")
        # print('DataCollator',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        return collated_batch


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser = BaseReaderSeq_KuaiRand.parse_data_args(parser)
    args = parser.parse_args()

    reader = BaseReaderSeq_KuaiRand(args)


    print(reader.data_df)

    # import pdb; pdb.set_trace()


    init_seed(SEED)
    train_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='train', shuffle=False, do_scale_image_to_01=True, image_resize=True, verbose=False)
    print(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=512, collate_fn=DataCollator())
    train_count = 0
    st = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    # print("start iter dataloader:",st)
    for batch in train_dataloader:
        for k, v in batch.items():
            print('Enumerate Data:',k, v.shape, v.dtype, v.max(), v.min())
            exit()
        # print("start iter dataloader:",st)
        # print("end first batch iter dataloader:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        train_count+=1
    print('count train batch', train_count)
    
    valid_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='dev', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)

    print(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, batch_size=512, collate_fn=DataCollator())
    valid_count = 0
    for batch in valid_dataloader:
        # for k, v in batch.items():
        #     print('Enumerate Data:',k, v.shape, v.dtype, v.max(), v.min())
            
        valid_count+=1
    print('count valid batch', valid_count)

    test_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    test_dataloader = DataLoader(test_dataset, batch_size=512, collate_fn=DataCollator())
    test_count = 0
    for batch in test_dataloader:
        # for k, v in batch.items():
            # print('Enumerate Data:',k, v.shape, v.dtype, v.max(), v.min())
        test_count+=1
    print('count test batch', test_count)
        

