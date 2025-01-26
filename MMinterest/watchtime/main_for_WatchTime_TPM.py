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
import math
import sys

logging.basicConfig(filename='Result_WatchTime.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S:%f', filemode='a')
logging.info(f"Command: {' '.join(sys.argv)}")

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

def prepareMemmap():
    mapping_path = "SegMM_photoidframeid2lineid.json"
    useridframeid2lineid = json.load(open(mapping_path))
    total_line = len(useridframeid2lineid)
    feat_memmap = np.memmap("/SegMM_feat_memmap.dat", dtype='float32', mode='r', shape=(total_line, 1024))
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

class TreeModelFastTest(nn.Module):
    def __init__(self, max_item, max_user, class_num, dropout, max_duration=200, emb_size=32):
        super(TreeModelFastTest, self).__init__()
        # Embedding layers for id
        self.max_item = max_item
        self.max_user = max_user
        self.item_embedding = nn.Embedding(max_item+1, emb_size)
        self.user_embedding = nn.Embedding(max_user+1, emb_size)
        # Embedding layer for duration
        self.duration_embedding = nn.Embedding(max_duration, emb_size)
        # Combined dense layers for final output
        # Fully connected layers
        self.fc1 = nn.Linear(emb_size*3, 128)
        # self.fc1 = nn.Linear(emb_size*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, class_num)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, user_id, item_id, duration, is_training):
        # Embedding
        if (item_id > self.max_item).any() or (user_id > self.max_user).any():
            import pdb; pdb.set_trace()
        item_emb = self.item_embedding(item_id.long())
        user_emb = self.user_embedding(user_id.long())
        # Embedding for duration
        duration_emb = self.duration_embedding(duration.long())
        # Concatenate all parts and pass through fully connected layers
        feas = torch.cat([item_emb, user_emb, duration_emb], dim=1)
        # feas = torch.cat([item_emb, user_emb], dim=1)
        # FC
        fc = F.relu(self.fc1(feas))
        if is_training:
            fc = self.dropout(fc)
        fc = F.relu(self.fc2(fc))
        if is_training:
            fc = self.dropout(fc)
        fc = F.relu(self.fc3(fc))
        
        logits = self.fc_out(fc)
        res = torch.sigmoid(logits) # probabilities
        return res

def get_percentile_of_playtime(df, wr_bucknum):
    if 'play_time_ms_x' in df.columns:
        play_time = np.array(df['play_time_ms_x']/5000)# TODO：这里之后可以处理超过40则为40
    else:
        play_time = np.array(df['playing_time_x']/5000)
    percen_value = np.percentile(play_time, np.linspace(0.0, 100.0, num=wr_bucknum + 1).astype(np.float32)).tolist()
    begins_tensor = torch.tensor(percen_value[:-1], dtype=torch.float32).unsqueeze(0)
    ends_tensor = torch.tensor(percen_value[1:], dtype=torch.float32).unsqueeze(0)
    return begins_tensor, ends_tensor

def label_encoding(tree_num_intervals, cmp_ratio, begins, ends, name="label_encoding"):
    label_dict = {}
    weight_dict = {}
    height = int(math.log2(tree_num_intervals))
    
    for i in range(height):
        for j in range(2 ** i):
            temp_ind = max(int(tree_num_intervals * 1.0 / (2 ** i) * j) - 1, 0)
            
            if j == 0:
                weight_temp = (cmp_ratio < begins[:, temp_ind]).float()
            else:
                weight_temp = (cmp_ratio < ends[:, temp_ind]).float()

            temp_ind = max(int(tree_num_intervals * 1.0 / (2 ** i) * (j + 1)) - 1, 0)
            weight_temp = (cmp_ratio < ends[:, temp_ind]).float() * weight_temp

            temp_ind = max(int(tree_num_intervals * (1.0 / (2 ** i) * j + 1.0 / (2 ** (i + 1)))) - 1, 0)
            label_temp = (cmp_ratio >= ends[:, temp_ind]).float()

            label_dict[1000 * i + j] = label_temp
            weight_dict[1000 * i + j] = weight_temp

    return label_dict, weight_dict

def get_label_encoding_loss(label_dict, weight_dict, label_encoding_predict, tree_num_intervals=32):
    auxiliary_loss = 0.0
    height = int(math.log2(tree_num_intervals))

    for i in range(height):
        for j in range(2 ** i):
            interval_label = label_dict[1000 * i + j]
            interval_weight = weight_dict[1000 * i + j]
            interval_preds = label_encoding_predict[:, 2 ** i - 1 + j]
            interval_loss = F.binary_cross_entropy_with_logits(interval_preds, interval_label, weight=interval_weight, reduction='sum')
            auxiliary_loss += interval_loss

    return auxiliary_loss / (tree_num_intervals - 1.0)

def get_encoded_playtime(label_encoding_predict, tree_num_intervals, begins, ends, name="encoded_playtime"):
    height = int(math.log2(tree_num_intervals))
    encoded_prob_list = []
    temp_encoded_playtime = (begins + ends) / 2.0  # bsx32
    encoded_playtime = temp_encoded_playtime
    
    for i in range(tree_num_intervals):
        temp = 0.0
        cur_code = 2 ** height - 1 + i
        for j in range(1, 1 + height):
            classifier_branch = cur_code % 2
            classifier_idx = (cur_code - 1) // 2
            cur_code = classifier_idx

            if classifier_branch == 1:
                temp += torch.log(1.0 - label_encoding_predict[:, classifier_idx] + 1e-5)
            else:
                temp += torch.log(label_encoding_predict[:, classifier_idx] + 1e-5)

        encoded_prob_list.append(temp)

    encoded_prob = torch.exp(torch.stack(encoded_prob_list, dim=1)).to(encoded_playtime.device)  # bs * tree_num_intervals
    encoded_playtime = torch.sum(encoded_playtime * encoded_prob, dim=-1, keepdim=True)

    e_x2 = torch.sum(torch.square(encoded_playtime) * encoded_prob, dim=-1, keepdim=True)
    square_of_e_x = torch.square(encoded_playtime)
    var = torch.sqrt(e_x2 - square_of_e_x)

    return encoded_playtime, torch.sum(var)

def tqm_loss(args, output, label, begins_tensor, ends_tensor, criterion):

    begins_tensor = begins_tensor.to(label.device)
    ends_tensor = ends_tensor.to(label.device)
    # Get encoded playtime and variance
    encoded_playtime, var = get_encoded_playtime(output, args.wr_bucknum, begins_tensor, ends_tensor)
    # Label encoding loss
    label_dict, weight_dict = label_encoding(args.wr_bucknum, label, begins_tensor, ends_tensor)
    loss_op = get_label_encoding_loss(label_dict, weight_dict, output, args.wr_bucknum)
    # MSE loss
    loss_op_mse = criterion(encoded_playtime.squeeze(), label)
    # Total loss
    loss = loss_op + loss_op_mse * args.mse_weight + var * args.var_weight
    return loss

def valid_model(args, valid_dataloader, model, criterion, begins_tensor, ends_tensor):
    model.eval()
    tmp_valid_loss = []
    for local_valid_step, valid_batch in enumerate(valid_dataloader):
        if args.debug:
            if local_valid_step>5: break
        valid_batch = {k: v.cuda() for k, v in valid_batch.items()}
        with torch.no_grad():
            output = model(user_id = valid_batch["user_id"], item_id = valid_batch["photo_id"], duration = valid_batch['duration'], is_training=False)
        label = (valid_batch['play_time']/40).float()
        label[label>1.0]=1.0
        loss = tqm_loss(args, output, label*40, begins_tensor, ends_tensor, criterion)
        # loss = tqm_loss(args, output, valid_batch['play_time'].float(), begins_tensor, ends_tensor, criterion)
        
        tmp_valid_loss.append(loss)
    return sum(tmp_valid_loss)/len(tmp_valid_loss)

def main(args):
    
    reader, train_dataloader, valid_dataloader, test_dataloader = load_data(args.dataname)

    begins_tensor, ends_tensor = get_percentile_of_playtime(reader.data_df['train'], args.wr_bucknum)
    
    model = TreeModelFastTest(reader.max_items,reader.max_users,args.wr_bucknum-1, dropout=0.2)
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    valid_loss_list, train_loss_list = [],[]
    for epoch in range(args.epochs):
        for local_step, batch in enumerate(train_dataloader):
            if args.debug:
                if local_step>5: break
            batch = {k: v.cuda() for k, v in batch.items()}
            model.train()
            optimizer.zero_grad()

            output = model(user_id = batch["user_id"], item_id = batch["photo_id"], duration = batch['duration'], is_training=True)
            label = (batch['play_time']/40).float()
            label[label>1.0]=1.0

            loss = tqm_loss(args, output, label*40, begins_tensor, ends_tensor, criterion)
            # loss = tqm_loss(args, output, batch['play_time'].float(), begins_tensor, ends_tensor, criterion)
            print(f'train loss: {loss}')
            train_loss_list.append(loss)

            loss.backward()
            optimizer.step()

            if (local_step+1)%args.valid_step == 0: # evaluation_step
                valid_loss = valid_model(args, valid_dataloader, model, criterion, begins_tensor, ends_tensor)
                print(f'train_loss: {sum(train_loss_list)/len(train_loss_list)} , valid_loss: {valid_loss}')
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
                output = model(user_id = test_batch["user_id"], item_id = test_batch["photo_id"], duration = test_batch['duration'], is_training=False)
            label = (test_batch['play_time']/40).float()
            label[label>1.0]=1.0
            vr_label = (label*40).squeeze() / test_batch['duration'].squeeze().float()
            
            encoded_playtime_test, _ = get_encoded_playtime(output, args.wr_bucknum, begins_tensor, ends_tensor)
            vr = torch.round(encoded_playtime_test).squeeze() / 40
            
            labels_all.append((label*40).cpu().numpy())
            preds_all.append(torch.round(encoded_playtime_test).numpy())
            preds_vr_all.append(vr.cpu().numpy())
            label_vr_all.append(vr_label.cpu().numpy())

        labels_all = np.concatenate(labels_all, axis=0)
        preds_all = np.concatenate(preds_all, axis=0)
        label_vr_all = np.concatenate(label_vr_all, axis=0)
        preds_vr_all = np.concatenate(preds_vr_all, axis=0)
        
        labels_all = labels_all.astype(int).squeeze()
        preds_all = preds_all.astype(int).squeeze()
        
        print('preds_all', preds_all)
        print('not 0', sum(preds_all>0))
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
    parser.add_argument('--wr_bucknum', type=int, default=16)
    parser.add_argument('--mse_weight', type=float, default=0.2)
    parser.add_argument('--var_weight', type=float, default=0.1)

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