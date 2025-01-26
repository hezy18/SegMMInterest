import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from model import MultiScaleTemporalDetr, SegFormerX, QueryBasedDecoder, main_eval_batch
# from utils import BaseReader, FrameDataset, DataCollator, DataLoader, LargeHDF5Cache
from utils import BaseReaderSeq_KuaiRand, FrameDatasetSeq_KuaiRand, DataCollator, DataLoader, LargeHDF5Cache
import numpy as np
import random
import os, json
import datetime
import glob, re, tqdm
from einops import einsum, repeat, rearrange, reduce
import pandas as pd
from sklearn.metrics import mean_squared_error

seed_value = 22
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
torch.manual_seed(seed_value) 
torch.cuda.manual_seed(seed_value)   
torch.cuda.manual_seed_all(seed_value)


train_videos_set=set()

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

        if len(args.eval_cold):
            train_videos_set.update(batch["photo_id"].cpu().tolist()) 


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
            # num_duration_pos[:duration]+=1
            num_view_duration_pos[duration-1,:viewlength]+=1
            if viewlength<40:
                num_view_duration_pos[duration-1,viewlength+1:]+=1
            num_leave_pos[viewlength]+=1
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

    prob_view_all = float(num_view_all / num_duration_all) 
    prob_view_pos = num_view_pos / count_case
    print(prob_view_pos)
    row_sums = num_view_duration_pos.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    prob_view_duration_pos = num_view_duration_pos / row_sums 
    prob_leave_pos = num_leave_pos / num_leave_pos.sum() 
    row_sums = num_leave_duration_pos.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1 
    prob_leave_duration_pos = num_leave_duration_pos / row_sums # shape: 40,41
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

def compute_final_result(results_list):
    final_result_dict={}
    if 'LeaveMSE' in results_list:
        view_lengths = results_list['view_lengths']
        pred_view_lengths = results_list['LeaveMSE']
        try:
            MSE = mean_squared_error(np.array(view_lengths), np.array(pred_view_lengths))
            print('LeaveMSE', MSE)
        except:
            print('fail to comput MSE')
            MSE=0
        final_result_dict['LeaveMSE'] = MSE

    for eval_type in results_list:
        print(eval_type)
        if eval_type == 'JaccardSim' or eval_type=='ProbAUC' or eval_type=='LeaveCTR' or eval_type=='LeaveCTR_view':
            avg_result = sum(results_list[eval_type])/len(results_list[eval_type])
            print(eval_type, avg_result)
            final_result_dict[eval_type] = avg_result
        elif eval_type=='TOP_K':
            continue
        elif eval_type=='LeaveMSE' or eval_type=='view_lengths' :
            continue
        else:
            avg_result = sum(results_list[eval_type])/len(results_list[eval_type])
            print(eval_type, avg_result)
            final_result_dict[eval_type] = avg_result
    return final_result_dict

def main(args, test_dataloader, statis_results, test_type):

    results_list={}
    if len(args.eval_cold):
        cold_results_list={}
        cold_count_inter = 0
        hot_results_list={}
        hot_count_inter = 0
    all_count_inter=0
    for eval_type in args.eval_type_list:
        results_list[eval_type]=[]
        results_list['view_lengths'] = []
        if len(args.eval_cold):
            cold_results_list[eval_type]=[]
            cold_results_list['view_lengths'] = []
            hot_results_list[eval_type]=[]
            hot_results_list['view_lengths'] = []
    if args.save_logits:
        save_logits_gt_seq_for_evaluate=torch.empty((0, 82))
    print("Starts Inference")
    all_count, cold_count, hot_count = 0,0,0
    for local_valid_step, test_batch in tqdm.tqdm(enumerate(test_dataloader)):
        if args.debug:
            if local_valid_step>2: break # To test code
        userID = test_batch["user_id"]
        itemID = test_batch["photo_id"]
        users = test_batch["user_id"]
        photos = test_batch["photo_id"]
        photo_ids = test_batch["photo_id"]
        bsz = test_batch["user_id"].shape[0]
        gt = test_batch["label"]
        durations = (gt!=-2).sum(dim=1, keepdim=True).numpy()

        # TODO: test here
        
        print(test_type)
        if test_type=='total_random':
            scores_tensor = torch.rand((bsz, 40))
        elif test_type=='all_same':
            scores_tensor = torch.ones((bsz, 40))
        elif test_type=='prob_view_all':
            probs = torch.full((bsz, 40),  statis_results[test_type]) 
            scores_tensor = torch.bernoulli(probs)
            # scores_tensor = torch.normal(probs, torch.ones_like(probs))
        elif test_type=='prob_view_pos':
            # normal_samples = np.clip(np.random.normal(statis_results[test_type], 0.1, size=(bsz,40)), 0, 1)
            # scores_tensor = torch.from_numpy(normal_samples).float()
            scores_tensor = torch.bernoulli(torch.from_numpy(statis_results[test_type]).repeat(bsz, 1))
        elif test_type=='prob_view_pos_static':
            scores_tensor = torch.from_numpy(statis_results['prob_view_pos']).repeat(bsz, 1)
        elif test_type=='prob_view_duration_pos':
            probs_numpy = np.zeros((bsz,40))
            prob_pos = statis_results[test_type]
            for i in range(bsz):
                duration = int(durations[i])
                probs_numpy[i,:] = prob_pos[duration-1,:]
            scores_tensor = torch.bernoulli(torch.from_numpy(probs_numpy))
        elif test_type=='prob_user_view_all':
            probs_numpy = np.zeros((bsz,40))
            prob_user_view_all = statis_results[test_type]
            for i in range(bsz):
                uid = users[i].item()
                if uid not in prob_user_view_all:
                    probs_numpy[i,:] = statis_results['prob_view_all']
                else:
                    probs_numpy[i,:] = prob_user_view_all[uid]
            # normal_samples = np.clip(np.random.normal(probs_numpy, 0.1, size=(bsz,40)), 0, 1)
            # scores_tensor = torch.from_numpy(normal_samples).float()
            scores_tensor = torch.bernoulli(torch.from_numpy(probs_numpy))
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
        elif test_type=='prob_user_view_pos_static':
            probs_numpy = np.zeros((bsz,40))
            prob_user_view_pos = statis_results['prob_user_view_pos']
            for i in range(bsz):
                uid = users[i].item()
                if uid not in prob_user_view_pos:
                    probs_numpy[i,:] = statis_results['prob_view_pos']
                else:
                    probs_numpy[i,:] = prob_user_view_pos[uid]
            scores_tensor = torch.from_numpy(probs_numpy)

        elif test_type=='num_item_view_duration_all':
            probs_numpy = np.zeros((bsz,40))
            num_item_view_duration_all = statis_results[test_type]
            for i in range(bsz):
                pid = photos[i].item()
                all_count+=1
                if pid not in num_item_view_duration_all:
                    cold_count+=1
                    probs_numpy[i,:] = statis_results['prob_view_all']
                else:
                    hot_count+=1
                    if num_item_view_duration_all[pid][1]==0:
                        probs_numpy[i,:] = 0
                    else:
                        probs_numpy[i,:] = num_item_view_duration_all[pid][0] / num_item_view_duration_all[pid][1]
            # normal_samples = np.clip(np.random.normal(probs_numpy, 0.1, size=(bsz,40)), 0, 1)
            # scores_tensor = torch.from_numpy(normal_samples).float()
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
        elif test_type=='num_item_view_duration_pos_static':
            probs_numpy = np.zeros((bsz,40))
            num_item_view_duration_pos = statis_results['num_item_view_duration_pos']
            for i in range(bsz):
                pid = photos[i].item()
                if pid not in num_item_view_duration_pos:
                    probs_numpy[i,:] = statis_results['prob_view_pos']
                else:
                    duration_pos = num_item_view_duration_pos[pid][1,:]
                    duration_pos[duration_pos==0]=1
                    probs_numpy[i,:] = num_item_view_duration_pos[pid][0,:] / duration_pos
            scores_tensor = torch.from_numpy(probs_numpy)
        else:
            return {}
            
        exposure_prob =  repeat(torch.tensor(args.exposure_prob), "L -> b L", b=bsz)
        logits = scores_tensor * exposure_prob
        pred_label = torch.where(logits>args.threshold, 1.0, 0.0)
        # import pdb; pdb.set_trace()
        if args.save_logits:
            combined = torch.cat((logits.cpu(), gt.cpu(), test_batch["user_id"].unsqueeze(-1).cpu(), test_batch["photo_id"].unsqueeze(-1).cpu()), dim=1)
            save_logits_gt_seq_for_evaluate = torch.cat((save_logits_gt_seq_for_evaluate, combined), dim=0)
        
        results_list = main_eval_batch(args, logits, gt, pred_label, results_list)
        
        all_count_inter += bsz
        if len(args.eval_cold):
            # cold_indices = [idx for idx, id_ in enumerate(test_batch["photo_id"].cpu().tolist()) if id_ not in train_videos_set]
            cold_indices = [idx for idx, id_ in enumerate(test_batch["photo_id"].cpu().tolist()) if id_ not in set(statis_results['num_item_view_duration_pos'].keys())]
            cold_count_inter+=len(cold_indices)
            cold_indices = torch.tensor(cold_indices, device=logits.device)
            cold_results_list = main_eval_batch(args, logits[cold_indices, :], gt[cold_indices, :], pred_label[cold_indices, :], cold_results_list, type='inference')
            
            hot_indices = [idx for idx, id_ in enumerate(test_batch["photo_id"].cpu().tolist()) if id_ in set(statis_results['num_item_view_duration_pos'].keys())]
            hot_count_inter+=len(hot_indices)
            hot_indices = torch.tensor(hot_indices, device=logits.device)
            hot_results_list = main_eval_batch(args, logits[hot_indices, :], gt[hot_indices, :], pred_label[hot_indices, :], hot_results_list, type='inference')

    
    if test_type=='num_item_view_duration_all':
        print('all_sample_count',all_count,'cold_sample_count',cold_count,'hot_sample_count',hot_count,'repeat_item_count',len(num_item_view_duration_all))
    
    final_result_dict = compute_final_result(results_list)
   
    if len(args.eval_cold):
        cold_final_result_dict = compute_final_result(cold_results_list)
        print(f"Test result on cold videos, {cold_final_result_dict}")
        print(f"all_inter:{all_count_inter}, cold_inter:{cold_count_inter}")

        hot_final_result_dict = compute_final_result(hot_results_list)
        print(f"Test result on hot videos, {hot_final_result_dict}")
        print(f"all_inter:{all_count_inter}, hot_inter:{hot_count_inter}")

        return final_result_dict, cold_final_result_dict, hot_final_result_dict
    
    else:
        return final_result_dict, None, None


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')

    parser.add_argument('--test_batch_size', type=int, default=1024)
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
    parser.add_argument('--eval_cold', type=str, default='test', choices=['','test', 'validtest'])
    
    parser = BaseReaderSeq_KuaiRand.parse_data_args(parser)
    args = parser.parse_args()

    exposure_prob = []
    if args.test_exposure_prob_type=='statistics':
        with open("data/KuaiRand_ExposureProb.json", "r") as f:
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
    if args.statistics:
        statis_results = statistics_dataset(reader)
        torch.save(statis_results, f'statis_results_KuaiRand_1219.pth')
    else:
        statis_results={}

    test_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, collate_fn=DataCollator())

    
    final = {}
    final_cold={}
    final_hot={}
    statis_results['total_random']=0
    statis_results['all_same']=0
    for test_type in ['total_random','all_same', 'prob_view_pos', # , 'prob_view_pos_static',
                     'prob_user_view_all', 'prob_user_view_pos', #, 'prob_user_view_pos_static',
                     'num_item_view_duration_all', 'num_item_view_duration_pos']: # ,'num_item_view_duration_pos_static']:
    # for test_type in ['num_item_view_duration_all']:
    # for test_type in statis_results:
        print(test_type)
        final_result_dict, cold_final_result_dict, hot_final_result_dict = main(args, test_dataloader, statis_results, test_type)
        if len(final_result_dict)==0:
            continue
        print(final_result_dict)
        final[test_type] = final_result_dict
        if cold_final_result_dict is not None:
            print(cold_final_result_dict)
            final_cold[test_type] = cold_final_result_dict
            print(hot_final_result_dict)
            final_hot[test_type] = hot_final_result_dict

    if len(args.eval_cold):
        with open(f'eval_results_KuaiRand/results_all_points/statistic_final_evalCold_{args.TOP_K_permutation}_{args.TOP_K_mask}_{seed_value}.json',"w") as fw:
            json.dump(final_cold, fw)
        all_keys = list(cold_final_result_dict.keys())
        rows = []
        for key, values in final_cold.items():
            row = {'key': key}
            row.update({k: values.get(k) for k in all_keys})
            rows.append(row)
            # df = df.append(row, ignore_index=True)
        df = pd.concat([pd.DataFrame([row], columns=['key']+all_keys) for row in rows], ignore_index=True)
        df.to_csv(f'eval_results_KuaiRand/results_all_points/statistic_final_evalCold_{args.TOP_K_permutation}_{args.TOP_K_mask}_{seed_value}.csv',index=False)

        with open(f'eval_results_KuaiRand/results_all_points/statistic_final_evalHot_{args.TOP_K_permutation}_{args.TOP_K_mask}_{seed_value}.json',"w") as fw:
            json.dump(final_hot, fw)
        all_keys = list(hot_final_result_dict.keys())
        rows = []
        for key, values in final_hot.items():
            row = {'key': key}
            row.update({k: values.get(k) for k in all_keys})
            rows.append(row)
            # df = df.append(row, ignore_index=True)
        df = pd.concat([pd.DataFrame([row], columns=['key']+all_keys) for row in rows], ignore_index=True)
        df.to_csv(f'eval_results_KuaiRand/results_all_points/statistic_final_evalHot_{args.TOP_K_permutation}_{args.TOP_K_mask}_{seed_value}.csv',index=False)


    with open(f'eval_results_KuaiRand/results_all_points/statistic_final_mask_{args.TOP_K_permutation}_{args.TOP_K_mask}_{seed_value}.json',"w") as fw:
        json.dump(final, fw)
    all_keys = list(final_result_dict.keys())
    # df = pd.DataFrame(columns=all_keys)
    rows = []
    for key, values in final.items():
        row = {'key': key}
        row.update({k: values.get(k) for k in all_keys})
        rows.append(row)
        # df = df.append(row, ignore_index=True)
    df = pd.concat([pd.DataFrame([row], columns=['key']+all_keys) for row in rows], ignore_index=True)
    df.to_csv(f'eval_results_KuaiRand/results_all_points/statistic_final_mask_{args.TOP_K_permutation}_{args.TOP_K_mask}_{seed_value}.csv',index=False)