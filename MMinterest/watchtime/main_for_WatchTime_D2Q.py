import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from utils import BaseReaderSeq_KuaiRand, FrameDatasetSeq_KuaiRand, DataCollator, DataLoader
from utils import BaseReaderSeq_SegMM, FrameDatasetSeq_SegMM, DataCollator, DataLoader
# from torch.optim import AdamW
import torch.optim as optim
import os
import numpy as np
import random
import wandb
from torch.nn.parallel import DistributedDataParallel
import json
import logging
from kn_util.nn_utils import CheckPointer
import torch.distributed as dist
import datetime
import matplotlib.pyplot as plt
import time
import glob, re, tqdm
from einops import einsum, repeat, rearrange, reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys

logging.basicConfig(filename='Result_WatchTime.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S:%f', filemode='a')
logging.info(f"Command: {' '.join(sys.argv)}")

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value) 

torch.manual_seed(seed_value) 
torch.cuda.manual_seed(seed_value) 
torch.cuda.manual_seed_all(seed_value) 

def prepareMemmap():
    mapping_path = "SegMM_photoidframeid2lineid.json"
    useridframeid2lineid = json.load(open(mapping_path))
    total_line = len(useridframeid2lineid)
    feat_memmap = np.memmap("SegMM_feat_memmap.dat", dtype='float32', mode='r', shape=(total_line, 1024))
    return useridframeid2lineid,feat_memmap

def load_data(dataname='KuaiRand'):
    print('Load Data')
    if dataname=='KuaiRand':
        reader = BaseReaderSeq_KuaiRand(args)
        train_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='train', shuffle=False, do_scale_image_to_01=True, image_resize=True, verbose=False)
        valid_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='dev', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
        test_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size, collate_fn=DataCollator())
        valid_dataloader = DataLoader(valid_dataset, args.valid_batch_size, collate_fn=DataCollator())
        test_dataloader = DataLoader(test_dataset, args.test_batch_size, collate_fn=DataCollator())
    elif dataname=='SegMM':
        reader = BaseReaderSeq_SegMM(args)
        useridframeid2lineid,feat_memmap = prepareMemmap()
        train_dataset = FrameDatasetSeq_SegMM(corpus=reader, lineid_map=useridframeid2lineid, feat_memmap=feat_memmap, phase='train', shuffle=False, do_scale_image_to_01=True, image_resize=True, verbose=False)
        valid_dataset = FrameDatasetSeq_SegMM(corpus=reader, lineid_map=useridframeid2lineid, feat_memmap=feat_memmap, phase='dev', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
        test_dataset = FrameDatasetSeq_SegMM(corpus=reader, lineid_map=useridframeid2lineid, feat_memmap=feat_memmap, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size, collate_fn=DataCollator())
        valid_dataloader = DataLoader(valid_dataset, args.valid_batch_size, collate_fn=DataCollator())
        test_dataloader = DataLoader(test_dataset, args.test_batch_size, collate_fn=DataCollator())
    return reader, train_dataloader, valid_dataloader, test_dataloader

class nnSwish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class D2QModel(nn.Module):
    def __init__(self, max_item, max_user, max_duration=200, 
                       emb_size=32, ):
        super(D2QModel, self).__init__()
        # Embedding layers for id
        self.max_item = max_item
        self.max_user = max_user
        self.item_embedding = nn.Embedding(max_item+1, emb_size)
        self.user_embedding = nn.Embedding(max_user+1, emb_size)
        # Embedding layer for duration
        self.duration_embedding = nn.Embedding(max_duration, emb_size)
        # Combined dense layers for final output
        self.swish = nnSwish()
        self.fc_layers = nn.Sequential(
            nn.Linear(3*emb_size, 512),self.swish,
            nn.Linear(512, 256),self.swish,
            nn.Linear(256, 128),self.swish,
            nn.Linear(128, 64),self.swish,
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    

    def forward(self, user_id, item_id, duration):
        # Embedding for categorical features

        if (item_id > self.max_item).any() or (user_id > self.max_user).any():
            import pdb; pdb.set_trace()
        item_emb = self.item_embedding(item_id.long())
        user_emb = self.user_embedding(user_id.long())
        # Embedding for duration
        duration_emb = self.duration_embedding(duration.long())
        # Concatenate all parts and pass through fully connected layers
        combined = torch.cat([item_emb, user_emb, duration_emb], dim=1)
        out = self.fc_layers(combined)
        return out


def valid_model(args, valid_dataloader, model,criterion):
    model.eval()
    tmp_valid_loss = []
    for local_valid_step, valid_batch in enumerate(valid_dataloader):
        valid_batch = {k: v.cuda() for k, v in valid_batch.items()}
        with torch.no_grad():
            output = model(user_id = valid_batch["user_id"], item_id = valid_batch["photo_id"], duration = valid_batch['duration'] )
        label = (valid_batch['play_time']/40).float()
        label[label>1.0]=1.0
        loss = criterion(output.squeeze(), label)
        tmp_valid_loss.append(loss)
    return sum(tmp_valid_loss)/len(tmp_valid_loss)


def main(args):
    
    reader, train_dataloader, valid_dataloader, test_dataloader = load_data(args.dataname)
    
    model = D2QModel(reader.max_items,reader.max_users)
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    
    valid_loss_list = []
    for epoch in range(args.epochs):
        for local_step, batch in enumerate(train_dataloader):
            batch = {k: v.cuda() for k, v in batch.items()}
            model.train()
            optimizer.zero_grad()
            output = model(user_id = batch["user_id"], item_id = batch["photo_id"], duration=batch['duration'] )
            label = (batch['play_time']/40).float()
            label[label>1.0]=1.0
            loss = criterion(output.squeeze(), label)
            print(f'train loss: {loss}')
            loss.backward()
            optimizer.step()

            if (local_step+1)%args.valid_step == 0: # evaluation_step
                valid_loss = valid_model(args, valid_dataloader, model, criterion)
                print(f'valid_loss: {valid_loss}')
                valid_loss_list.append(valid_loss)
                if args.early_stop > 0: 
                    if len(valid_loss_list) > args.early_stop: 
                        lst = valid_loss_list[-args.early_stop:]
                        if all(x <= y for x, y in zip([lst[0]]*(len(lst)-1), lst[1:])): 
                            break
                    if len(valid_loss_list) - valid_loss_list.index(min(valid_loss_list)) > args.early_stop:
                        break
        
    if args.test_model:
        labels_all = []
        preds_all = []
        preds_vr_all = []
        label_vr_all = []
        model.eval()
        for local_test_step, test_batch in tqdm.tqdm(enumerate(test_dataloader)):
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            with torch.no_grad():
                output = model(user_id = test_batch["user_id"], item_id = test_batch["photo_id"], duration = test_batch['duration'] )
            label = (test_batch['play_time']/40).float()
            label[label>1.0]=1.0
            # vr_label = (label*40).squeeze() / test_batch['duration'].squeeze().float()
            # loss = criterion(output.squeeze(), label)
            # vr = torch.round((output*40)).float().squeeze() / test_batch['duration'].float()

            labels_all.append((label*40).int().squeeze().cpu().numpy())
            preds_all.append(torch.round((output*40)).squeeze().cpu().numpy())
            preds_vr_all.append(output.squeeze().cpu().numpy())
            label_vr_all.append(label.squeeze().cpu().numpy())
        
        labels_all = np.concatenate(labels_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        label_vr_all = np.concatenate(label_vr_all, axis=0)
        preds_vr_all = np.concatenate(preds_vr_all, axis=0)
        HR1 = float(np.sum(labels_all == preds_all) / np.prod(labels_all.shape))
        import pandas as pd
        df = pd.DataFrame({'labels': labels_all, 'preds': preds_all})
        df = df.dropna()
        labels_all = df['labels'].values
        preds_all = df['preds'].values
        mae = mean_absolute_error(labels_all, preds_all)
        print(f"HR1: {HR1}, mae: {mae}")
        logging.info(f"HR1: {HR1}, mae: {mae}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')

    parser.add_argument('--dataname', type=str, default='KuaiRand')
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--valid_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4) 
    parser.add_argument('--early_stop', type=int, default=20, help='The number of epochs when dev results drop continuously.')
    parser.add_argument('--test_model', type=int, default=1)
    parser.add_argument('--valid_step', type=int, default=30)

    args = parser.parse_args()

    if args.dataname == 'KuaiRand':
        parser = BaseReaderSeq_KuaiRand.parse_data_args(parser)
    elif args.dataname == 'SegMM':
        parser = BaseReaderSeq_SegMM.parse_data_args(parser)
    else:
        raise ValueError(f"Unsupported dataname: {args.dataname}")
    
    args = parser.parse_args() 
    logging.info(f'Args: {vars(args)}')
    
    print(args)
    main(args)