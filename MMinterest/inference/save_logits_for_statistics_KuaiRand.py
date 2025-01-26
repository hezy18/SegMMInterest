import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from model import MultiScaleTemporalDetr, SegFormerX, QueryBasedDecoder, main_eval_batch
from utils import BaseReader, FrameDataset, DataCollator, DataLoader, LargeHDF5Cache
from utils import BaseReaderSeq_KuaiRand, FrameDatasetSeq_KuaiRand, DataCollator, DataLoader, LargeHDF5Cache
import numpy as np
import random
import os, json
import datetime
import glob, re, tqdm
from einops import einsum, repeat, rearrange, reduce
import pandas as pd

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value) 
torch.manual_seed(seed_value)    
torch.cuda.manual_seed(seed_value)   
torch.cuda.manual_seed_all(seed_value) 

def statistics_dataset(reader):
    reader.data_df['train_dev'] = pd.concat([reader.data_df[key] for key in ['train', 'dev']])
    train_dev_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='train_dev', shuffle=False, do_scale_image_to_01=True, image_resize=True, verbose=False)
    statis_dataloader = DataLoader(train_dev_dataset, 1024, collate_fn=DataCollator())

    num_view_all, num_duration_all = 0,0
    num_view_pos, num_duration_pos = np.zeros((40)),np.zeros((40))
    num_view_duration_pos = np.zeros((40,40))
    num_leave_pos = np.zeros((41))
    num_leave_duration_pos = np.zeros((40,41))
    photo_view_duration_all = {} # pid:(view_count,duration_count)
    photo_view_duration_pos = {} # pid:(view_count,duration_count)*40
    user_view_duration_all = {}
    user_view_duration_pos = {}


    count_case=0

    for local_step, batch in tqdm.tqdm(enumerate(statis_dataloader)):
        photo_ids = batch["photo_id"]
        user_ids = batch["user_id"]
        bsz = user_ids.shape[0]
        gt = batch["label"]

        if args.debug:
            if local_step>3: break
        num_view_all += (gt==1).sum().item()
        num_duration_all += (gt!=-2).sum().item()


        view_lengths = (gt==1).sum(dim=1, keepdim=True).numpy()
        durations = (gt!=-2).sum(dim=1, keepdim=True).numpy()
        for pid, uid, duration, viewlength, label in zip(photo_ids,user_ids, durations, view_lengths, gt):
            viewlength = int(viewlength)
            duration = int(duration)
            uid = uid.item()
            pid = pid.item()
            count_case+=1
            num_view_pos[:viewlength]+=1
            if viewlength<40:
                num_view_pos[viewlength+1:]+=1
            num_view_duration_pos[duration-1,:viewlength]+=1
            if viewlength<40:
                num_view_duration_pos[duration-1,viewlength+1:]+=1
            num_leave_duration_pos[duration-1,viewlength]+=1
            if pid not in photo_view_duration_all:
                photo_view_duration_all[pid]=[0,0]
                photo_view_duration_pos[pid]=np.zeros((2,40))
            photo_view_duration_all[pid][0]+=viewlength
            photo_view_duration_all[pid][1]+=duration
            photo_view_duration_pos[pid][0,:viewlength]+=1
            if viewlength<40:
                photo_view_duration_pos[pid][0,viewlength+1:]+=1
            photo_view_duration_pos[pid][1,:]+=1
            if uid not in user_view_duration_all:
                user_view_duration_all[uid]=[0,0]
                user_view_duration_pos[uid]=np.zeros((2,40))
            user_view_duration_all[uid][0]+=viewlength
            user_view_duration_all[uid][1]+=duration
            user_view_duration_pos[uid][0,:viewlength]+=1
            if viewlength<40:
                user_view_duration_pos[uid][0,viewlength+1:]+=1
            user_view_duration_pos[uid][1,:]+=1

    
    prob_view_all = float(num_view_all / num_duration_all) #
    prob_view_pos = num_view_pos / count_case
    print(prob_view_pos)
    row_sums = num_view_duration_pos.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    prob_view_duration_pos = num_view_duration_pos / row_sums 
    prob_leave_pos = num_leave_pos / num_leave_pos.sum() 
    row_sums = num_leave_duration_pos.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    prob_leave_duration_pos = num_leave_duration_pos / row_sums 
    print(prob_leave_duration_pos)
    prob_user_view_all,prob_user_view_pos={},{}
    for uid in user_view_duration_all:
        if user_view_duration_all[uid][1]==0:
            prob_user_view_all[uid] = 0
        else:
            prob_user_view_all[uid] = user_view_duration_all[uid][0] / user_view_duration_all[uid][1]
        duration_pos = user_view_duration_pos[uid][1,:]
        # duration_pos[duration_pos==0]=1
        prob_user_view_pos[uid] = user_view_duration_pos[uid][0,:] / duration_pos
    
    # import pdb; pdb.set_trace()
    return {'prob_view_all':prob_view_all, 'prob_view_pos':prob_view_pos, 'prob_view_duration_pos':prob_view_duration_pos, 
            'prob_leave_pos':prob_leave_pos, 'prob_leave_duration_pos':prob_leave_duration_pos,
            'prob_user_view_all':prob_user_view_all, 'prob_user_view_pos':prob_user_view_pos,
            'num_item_view_duration_all': photo_view_duration_all, 'num_item_view_duration_pos':photo_view_duration_pos}


def main(args, train_dataloader, valid_dataloader, test_dataloader, statis_results, test_type):
    test_logits = {}

    for split in ["train", "dev", "test"]:
        print(f"Starts Inference on {split}")
        if split=="train":
            dataloader = train_dataloader
        elif split=="dev":
            dataloader = valid_dataloader
        elif split=="test":
            dataloader = test_dataloader
    
        for local_valid_step, test_batch in tqdm.tqdm(enumerate(dataloader)):
            if args.debug:
                if local_valid_step>2: break # To test code
            userID = test_batch["user_id"]
            itemID = test_batch["user_id"]
            users = test_batch["user_id"]
            photos = test_batch["photo_id"]
            photo_ids = test_batch["photo_id"]
            bsz = test_batch["user_id"].shape[0]
            gt = test_batch["label"]
            durations = (gt!=-2).sum(dim=1, keepdim=True).numpy()

            print(test_type)
            if test_type=='all_same':
                scores_tensor = torch.ones((bsz, 40))
            elif test_type=='prob_view_pos':
                scores_tensor = torch.bernoulli(torch.from_numpy(statis_results[test_type]).repeat(bsz, 1))
            elif test_type=='prob_user_view_pos':
                probs_numpy = np.zeros((bsz,40))
                prob_user_view_pos = statis_results[test_type]
                for i in range(bsz):
                    uid = users[i].item()
                    if uid not in prob_user_view_pos:
                        probs_numpy[i,:] = statis_results['prob_view_pos']
                    else:
                        probs_numpy[i,:] = prob_user_view_pos[uid]
                scores_tensor = torch.bernoulli(torch.from_numpy(probs_numpy))
            elif test_type=='num_item_view_duration_pos':
                probs_numpy = np.zeros((bsz,40))
                num_item_view_duration_pos = statis_results[test_type]
                for i in range(bsz):
                    pid = photos[i].item()
                    if pid not in num_item_view_duration_pos:
                        probs_numpy[i,:] = statis_results['prob_view_pos']
                    else:
                        duration_pos = num_item_view_duration_pos[pid][1,:]
                        duration_pos[duration_pos==0]=1
                        probs_numpy[i,:] = num_item_view_duration_pos[pid][0,:] / duration_pos
                scores_tensor = torch.bernoulli(torch.from_numpy(probs_numpy))
            else:
                return {}
            
            exposure_prob =  repeat(torch.tensor(args.exposure_prob), "L -> b L", b=bsz)
            logits = scores_tensor * exposure_prob
        
            user_id_list = test_batch["user_id"].cpu().tolist()
            photo_id_list = test_batch["photo_id"].cpu().tolist()
            time_ms_list = test_batch["time_ms"].cpu().tolist()
            for uid, pid, time_ms, logit in zip(user_id_list, photo_id_list, time_ms_list, logits):
                test_logits[f"{uid}-{pid}-{time_ms}"] = logit.cpu().detach().tolist()
             
    save_dir = 'saved_logits/KuaiRand'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, 'statistics_' + test_type + '.json')
    with open(save_path, "w") as fw:
        json.dump(test_logits, fw)
    
    torch.save(test_logits, save_path.replace("json", "pth"))
    print(f"Done {test_type}")
    print(f"Save to {save_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')

    parser.add_argument('--test_batch_size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--result_path',type=str, default='')
    parser.add_argument('--test_exposure_prob_type',type=str, default='ones', choices=['ones', 'statistics'])
    parser.add_argument('--save_logits', type=int, default=0)
    parser.add_argument('--eval_type_list', type=str, default='JaccardSim,ProbAUC,LeaveMSE,LeaveCTR,LeaveCTR_view,TOP_K', help='combination of JaccardSim, ProbAUC, LeaveMSE, split by ,')
    parser.add_argument('--draw_case', type=int, default=0)
    parser.add_argument('--statistics', type=int, default=1)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--TOP_K_permutation', type=int, default=1)
    parser.add_argument('--TOP_K_mask', type=int, default=0)

    parser = BaseReaderSeq_KuaiRand.parse_data_args(parser)
    args = parser.parse_args()
    exposure_prob = []
    if args.test_exposure_prob_type=='statistics':
        with open("data/ExposureProb.json", "r") as f:
            probs = json.load(f)
            for idx in probs:
                exposure_prob.append(probs[idx])
    elif args.test_exposure_prob_type=='ones':
        exposure_prob = [1.0] * 40
    else:
        raise KeyError('test_exposure_prob_type must be statistics or ones')
    args.exposure_prob = exposure_prob

    if ',' in args.eval_type_list:
        args.eval_type_list = [item.strip() for item in args.eval_type_list.split(',')]
    else:
        args.eval_type_list = [args.eval_type_list.strip()]

    print('start FrameDataset')
    reader = BaseReaderSeq_KuaiRand(args)
    # useridframeid2lineid,feat_memmap = prepareMemmap()
    if args.statistics:
        statis_results = statistics_dataset(reader)
    else:
        statis_results={}

    train_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='train', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    valid_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='dev', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    train_dataloader = DataLoader(train_dataset, args.test_batch_size, collate_fn=DataCollator())
    valid_dataloader = DataLoader(valid_dataset, args.test_batch_size, collate_fn=DataCollator())
    
    test_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, collate_fn=DataCollator())

    statis_results['all_same']=0
    for test_type in ['all_same', 'prob_view_pos', 'prob_user_view_pos', 'num_item_view_duration_pos']:
    # for test_type in ['prob_user_view_pos']:

        print(test_type)
        main(args, train_dataloader, valid_dataloader, test_dataloader, statis_results, test_type)