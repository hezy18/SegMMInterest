import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from model import MultiScaleTemporalDetrLeaveFocal, SegFormerX, QueryBasedDecoder, main_eval_batch, TOP_K_leave, TOP_K_leave_mask
from utils import BaseReaderSeq_KuaiRand, FrameDatasetSeq_KuaiRand, DataCollator, DataLoader
from torch.optim import AdamW
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

logger = logging.getLogger(__name__)
seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)

torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)

def load_data():
    print('Load Data')
    reader = BaseReaderSeq_KuaiRand(args)
    train_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='train', shuffle=False, do_scale_image_to_01=True, image_resize=True, verbose=False)
    valid_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='dev', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    test_dataset = FrameDatasetSeq_KuaiRand(corpus=reader, phase='test', shuffle=False, do_scale_image_to_01=True, image_resize=True,verbose=False)
    train_dataloader = DataLoader(train_dataset, args.train_batch_size, collate_fn=DataCollator())
    valid_dataloader = DataLoader(valid_dataset, args.valid_batch_size, collate_fn=DataCollator())
    test_dataloader = DataLoader(test_dataset, args.test_batch_size, collate_fn=DataCollator())
    return reader, train_dataloader, valid_dataloader, test_dataloader

def init_model(args, reader):
    print('Initing Model')
    frame_pooler = nn.Identity()
    if args.input_type['user']=='both' or  args.input_type['photo']=='both':
        if args.input_type['user']=='both':
            user_id_max1=-1 # 先image
            max_usr_len1=100
            user_id_max2=reader.n_users # 后id
            max_usr_len2=1
        elif args.input_type['user']=='id':
            user_id_max1=reader.n_users
            max_usr_len1=1
            user_id_max2=reader.n_users
            max_usr_len2=1
        elif args.input_type['user']=='image':
            user_id_max1=-1
            max_usr_len1=100
            user_id_max2=-1
            max_usr_len2=100
        if args.input_type['photo']=='both':
            video_id_max1 = -1
            video_id_max2 =reader.n_items
        elif args.input_type['photo']=='id':
            video_id_max1 = reader.n_items
            video_id_max2 = reader.n_items
        elif args.input_type['photo']=='image':
            video_id_max1 = -1
            video_id_max2 = -1
        backbone1 = SegFormerX(d_model_in=args.d_model,
                        d_model_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        num_head_lvls=[args.nhead] * args.num_layers_enc,#eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                        ff_dim_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        input_vid_dim=1024,input_usr_dim=1024,
                        max_vid_len=40, max_usr_len=max_usr_len1,
                        sr_ratio_lvls=[1] * args.num_layers_enc, use_patch_merge=[False] * args.num_layers_enc,
                        output_layers=[-1], model_cfg = args,
                        user_id_max = user_id_max1, video_id_max = video_id_max1,use_pe = args.use_pe)
        backbone2 = SegFormerX(d_model_in=args.d_model,
                        d_model_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        num_head_lvls=[args.nhead] * args.num_layers_enc,#eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                        ff_dim_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        input_vid_dim=1024,input_usr_dim=1024,
                        max_vid_len=40, max_usr_len=max_usr_len2,
                        sr_ratio_lvls=[1] * args.num_layers_enc, use_patch_merge=[False] * args.num_layers_enc,
                        output_layers=[-1], model_cfg = args,
                        user_id_max = user_id_max2, video_id_max = video_id_max2,use_pe = args.use_pe)
        model = MultiScaleTemporalDetrLeaveFocal(backbone1, backbone2, None, frame_pooler, args)
    else:
        if args.input_type['user']=='id':
            user_id_max1=reader.n_users
            max_usr_len1=1
        elif args.input_type['user']=='image':
            user_id_max1=-1
            max_usr_len1=100
        if args.input_type['photo']=='id':
            video_id_max1 = reader.n_items
        elif args.input_type['photo']=='image':
            video_id_max1 = -1
        backbone1 = SegFormerX(d_model_in=args.d_model,
                        d_model_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        num_head_lvls=[args.nhead] * args.num_layers_enc,#eval_str(s="[${model_cfg.nhead}] * ${model_cfg.num_layers_enc}"),
                        ff_dim_lvls=[args.d_model] * args.num_layers_enc,#eval_str(s="[${model_cfg.d_model}] * ${model_cfg.num_layers_enc}"),
                        input_vid_dim=1024,input_usr_dim=1024,
                        max_vid_len=40, max_usr_len=max_usr_len1,
                        sr_ratio_lvls=[1] * args.num_layers_enc, use_patch_merge=[False] * args.num_layers_enc,
                        output_layers=[-1], model_cfg = args,
                        user_id_max = user_id_max1, video_id_max = video_id_max1,use_pe = args.use_pe)
        
    
        model = MultiScaleTemporalDetrLeaveFocal(backbone1, None, None, frame_pooler, args)
    return model

def valid_model(args, valid_dataloader, total_valid_loss_metrics, model, logger):
    model.eval()
    tmp_valid_loss_metrics = {}
    for loss_metric in total_valid_loss_metrics:
        tmp_valid_loss_metrics[loss_metric] = []
    
    valid_equal_num=0
    valid_all_num=0
    for local_valid_step, valid_batch in enumerate(valid_dataloader):
        if args.debug: 
            if local_valid_step>3:break # To test code
        valid_batch = {k: v.cuda() for k, v in valid_batch.items()}
        # print('valid usr_feat shape', usr_feat.shape)
        valid_output = model(usr_image = valid_batch["user_id"], usr_id = valid_batch["user_id"],usr_mask = valid_batch["user_mask"], 
                    vid_image = valid_batch["photo_id"], vid_id = valid_batch["photo_id"], vid_mask=valid_batch["photo_mask"], 
                    gt=valid_batch["label"], mode="train")
        bsz = valid_batch["user_id"].shape[0]
        exposure_prob =  repeat(torch.tensor(args.exposure_prob).cuda(), "L -> b L", b=bsz)
        interests = torch.sigmoid(valid_output["logits"])  * exposure_prob
        gt = valid_output["gt"]
        view_lengths = (gt == 1).sum(dim=1, keepdim=True).cpu().numpy()
        mask_batch = (gt != -2).cpu().detach().numpy()
        if args.count_view_completion:
            view_lengths_1 = (gt == 1).sum(dim=1, keepdim=True)
            durations_1 = (gt != -2).sum(dim=1, keepdim=True)
            valid_equal_num += (view_lengths_1 == durations_1).sum().item()
            valid_all_num+=bsz
        
        loss = valid_output["loss"]
        # # TODO:debug----
        # pred_label = torch.where(interests>args.threshold, 1.0, 0.0)
        # main_eval_batch(args, interests, gt, pred_label, results_list={},type='inference')
        # # TODO: --------
        if args.TOP_K_mask:
            evaluations = TOP_K_leave_mask(interests.cpu().detach().numpy(), view_lengths, mask_batch, permutation = args.TOP_K_permutation)
        else:
            evaluations = TOP_K_leave(interests.cpu().detach().numpy(), view_lengths, mask_batch, permutation = args.TOP_K_permutation)
        tmp_valid_loss_metrics['valid_loss'].append(float(loss.item()))
        for loss_metric in tmp_valid_loss_metrics:
            # print(loss_metric)
            if loss_metric in valid_output:
                tmp_valid_loss_metrics[loss_metric].append(float(valid_output[loss_metric].item()))
            elif loss_metric in evaluations:
                tmp_valid_loss_metrics[loss_metric].append(float(evaluations[loss_metric]))
        # tmp_main_evaluation_list.append(evaluations[args.main_metrics])
        logger.info({k: v for k, v in valid_output.items() if k not in ['gt', 'logits']})
        # tmp_valid_loss_list.append(loss.item())

    for loss_metric in tmp_valid_loss_metrics:
        if len(tmp_valid_loss_metrics[loss_metric])!=0:
            total_valid_loss_metrics[loss_metric].append(sum(tmp_valid_loss_metrics[loss_metric])/len(tmp_valid_loss_metrics[loss_metric]))
    if args.record_train_detail: 
        record_dict = {'valid_loss':loss.item(),'valid_gt':gt.cpu().detach(),'valid_interests':interests.cpu().detach()}
        return total_valid_loss_metrics, record_dict, valid_equal_num,valid_all_num
    else:
        return total_valid_loss_metrics, {}, valid_equal_num,valid_all_num

def compute_final_result(results_list,sample_count=None):
    
    final_result_dict={}
    if 'LeaveMSE' in results_list:
        view_lengths = np.array(results_list['view_lengths'])
        duration_lengths = np.array(results_list['duration_lengths'])
        pred_view_lengths = np.array(results_list['LeaveMSE'])
        mse = mean_squared_error(view_lengths, pred_view_lengths)
        mae = mean_absolute_error(view_lengths, pred_view_lengths)
        print('LeaveMSE', mse, mae)
        final_result_dict['LeaveMSE'] = (mse, mae)
    
    if 'TOP1MSE' in results_list:
        view_lengths = np.array(results_list['view_lengths'])
        duration_lengths = np.array(results_list['duration_lengths'])
        pred_view_lengths = np.concatenate(results_list['TOP1MSE'])
        mse = mean_squared_error(view_lengths, pred_view_lengths)
        mae = mean_absolute_error(view_lengths, pred_view_lengths)
        print('TOP1MSE', mse, mae)
        final_result_dict['TOP1MSE'] = (mse, mae)
    
    if 'MAES' in results_list and sample_count!=None:
        final_result_dict['MAES'] = []
        for i in range(len(results_list['MAES'])):
            final_result_dict['MAES'].append(results_list['MAES'][i] / sample_count)
    if 'pred_leave' in results_list:
        final_result_dict['pred_leave']={}
        view_lengths = np.array(results_list['view_lengths'])
        pred_view_lengths = np.concatenate(results_list['pred_leave'])
        mse = mean_squared_error(view_lengths, pred_view_lengths)
        mae = mean_absolute_error(view_lengths, pred_view_lengths)
        final_result_dict['pred_leave'] = (mse, mae)
    
    for eval_type in results_list:
        if eval_type == 'JaccardSim' or eval_type=='ProbAUC' or eval_type=='LeaveCTR' or eval_type=='LeaveCTR_view':
            avg_result = sum(results_list[eval_type])/len(results_list[eval_type])
            print(eval_type, avg_result)
            final_result_dict[eval_type] = avg_result
        elif eval_type=='TOP_K':
            continue
        elif eval_type in ['LeaveMSE', 'view_lengths', 'TOP1MSE', 'MAES', 'pred_leave']:
            continue
        else:
            avg_result = sum(results_list[eval_type])/len(results_list[eval_type])
            print(eval_type, avg_result)
            final_result_dict[eval_type] = avg_result
    return final_result_dict

def main(args):
    
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    param_dir = f'{args.num_layers_enc}_{args.exposure_prob_type}_{args.learning_rate}_{args.weight_decay}_{args.learnable_bias}_{args.loss_type}_{args.loss_weight_interestBPR}_{args.user_input_type}_{args.photo_input_type}_{args.mask_loss}_{args.use_pe}_{args.fusion_heads}_earlystop_focal'
    ckpt = CheckPointer("main_metric", os.path.join(args.ckpt_dir,param_dir), mode="max", cur_time=cur_time)
    
    reader, train_dataloader, valid_dataloader, test_dataloader = load_data()
    
    model = init_model(args, reader)
    model = model.cuda()
    if len(args.eval_cold):
        train_videos_set=set()
    param_dict = model.parameters()
    optimizer = AdamW(param_dict, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)
    total_train_loss = []
    total_valid_loss_metrics = {'train_loss':[], 'valid_loss':[]}
    for loss_type in args.loss_type_list:
        total_valid_loss_metrics[loss_type] = []
    for eval_type in args.eval_type_list:
        if eval_type == 'TOP_K':
            for k in [1,3,5,10]:
                for metric in ['HR','NDCG']:
                    key = '{}@{}'.format(metric, k)
                    total_valid_loss_metrics[key] = []
        else:
            total_valid_loss_metrics[eval_type] = []
    global_step = 0
    if args.record_train_detail:
        record_dict_list = []

    logging.basicConfig(filename=f'logs_KuaiRand/{cur_time}_{args.num_layers_enc}_{args.exposure_prob_type}_{args.learning_rate}_{args.learnable_bias}_{args.loss_type}_{args.loss_weight}_{args.input_type}_{args.use_pe}_{args.fusion_heads}_earlystop_focal.txt', format='%(asctime)s %(levelname)s {%(module)s} [%(funcName)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S:%f', level=logging.INFO)
    logger.info(args)
    print('Evaluation Before Training')
    total_valid_loss_metrics['train_loss'].append(0.0)
    total_valid_loss_metrics, record_dict, _ , _ = valid_model(args, valid_dataloader, total_valid_loss_metrics, model, logger)
    if args.record_train_detail:
        print('valid_loss',record_dict['valid_loss'])
        print(record_dict['valid_interests'])
    if args.count_view_completion:
        equal_num = {}
    print('Start Training')
    stop_flag = False
    for epoch in range(args.epochs):
        if stop_flag:
            break
        logger.info(f"Epoch {epoch} starts")
        epoch_st = time.time()
        if args.count_view_completion:
            equal_num['train']=0
            equal_num['train_all']=0
        for local_step, batch in enumerate(train_dataloader):
            train_step_st = time.time()
            if args.debug: 
                if local_step>3:break # To test code
            optimizer.zero_grad()
            model.train()
            batch = {k: v.cuda() for k, v in batch.items()}
            train_output = model(usr_image = batch["user_id"], usr_id = batch["user_id"], usr_mask = batch["user_mask"], 
                        vid_image = batch["photo_id"], vid_id = batch["photo_id"], vid_mask = batch["photo_mask"], 
                        gt=batch["label"], mode="train")
            loss = train_output["loss"]
            if args.count_view_completion:
                view_lengths = (batch["label"] == 1).sum(dim=1, keepdim=True)
                durations = (batch["label"] != -2).sum(dim=1, keepdim=True)
                equal_num['train'] += (view_lengths == durations).sum().item()
                equal_num['train_all']+=batch["user"].shape[0]
            
            if len(args.eval_cold):
                train_videos_set.update(batch["photo_id"].cpu().tolist()) 

            loss.backward()
            logger.info({k: v for k, v in train_output.items() if k not in ['gt', 'logits']})
            grad_norm = nn.utils.clip_grad_norm_(param_dict, 10.0).item()
            optimizer.step()
            total_train_loss.append(loss.item())
            global_step += 1
            train_step_et = time.time()
            if (local_step+1)%args.logging_step == 0: # logging_step
                logger.info(f"Train_loss: {loss.item()}, Global_step: {global_step}, training_time_use_per_step: {train_step_et-train_step_st}")
            if (local_step+1)%args.valid_step == 0: # evaluation_step
                print(local_step, 'start validation')
                total_valid_loss_metrics['train_loss'].append(float(loss.item()))
                valid_st = time.time()
                total_valid_loss_metrics, record_dict, valid_equal_num, valid_all_num = valid_model(args, valid_dataloader, total_valid_loss_metrics, model, logger)
                if args.count_view_completion:
                    equal_num['valid'] = valid_equal_num
                    equal_num['valid_all'] = valid_all_num
                valid_et = time.time()
                if args.record_train_detail:
                    train_loss = loss.item()
                    bsz = train_output["gt"].shape[0]
                    train_gt = train_output["gt"].cpu().detach()
                    exposure_prob =  repeat(torch.tensor(args.exposure_prob).cuda(), "L -> b L", b=bsz)
                    train_interests = (torch.sigmoid(train_output["logits"])  * exposure_prob).cpu().detach()
                    record_dict.update({'epoch':epoch, 'step':local_step, 'train_loss':train_loss, 'train_gt':train_gt,'train_interests':train_interests})
                    record_dict_list.append(record_dict)
                    print('train_loss',record_dict['train_loss'])
                    print(train_interests)
                    print('valid_loss',record_dict['valid_loss'])
                    print(record_dict['valid_interests'])
                    # import pdb; pdb.set_trace()
                        
                avg_main_evaluation = total_valid_loss_metrics[args.main_metrics][-1]
                logger.info(f"Valid_loss: {total_valid_loss_metrics['valid_loss'][-1]}, {args.main_metrics}: {avg_main_evaluation}, Global_step: {global_step}, Evaluation_time_use: {valid_et-valid_st}")
                
                # save ckpt
                save_model = model
                better = ckpt.save_checkpoint(model=save_model, optimizer=optimizer, num_epochs=epoch, metric_vals={"main_metric": avg_main_evaluation})
                
                total_valid_metric = total_valid_loss_metrics[args.main_metrics]
                if args.early_stop > 0: 
                    if len(total_valid_metric) > args.early_stop: 
                        lst = total_valid_metric[-args.early_stop:]
                        if all(x >= y for x, y in zip([lst[0]]*(len(lst)-1), lst[1:])): 
                            stop_flag = True
                            break
                    if len(total_valid_metric) - total_valid_metric.index(max(total_valid_metric)) > args.early_stop:
                        stop_flag = True
                        break
        epoch_et = time.time()
        logger.info(f"Epoch: {epoch}, Epoch Loss: {sum(total_train_loss)/len(total_train_loss)}, Train_time_use: {epoch_et - epoch_st}")
    if stop_flag:
        logger.info("Early stop at %d based on dev result." % (epoch + 1))

    if args.record_train_detail:
        with open(f"DebugAndCheck/KuaiRand/{param_dir}_{args.TOP_K_permutation}_{args.TOP_K_mask}_{cur_time}.json", "w") as fw:
            json.dump(total_valid_loss_metrics, fw)
        torch.save(record_dict_list, f"DebugAndCheck/KuaiRand/{param_dir}__{args.TOP_K_permutation}_{args.TOP_K_mask}_{cur_time}_record_logit_gt.pth")

    # test metric cal
    # run inference.py
    if args.test_model:
        load_dict = ckpt.load_checkpoint(model,optimizer,mode='best')
        model.load_state_dict(load_dict["model"])
        model = model.cuda()
        model.eval()
        results_list={}
        if len(args.eval_cold):
            cold_results_list={}
            cold_count_inter = 0
            hot_results_list={}
            hot_count_inter = 0
        for eval_type in args.eval_type_list:
            results_list[eval_type]=[]
            results_list['view_lengths'] = []
            results_list['duration_lengths'] = []
            results_list['MAES'] = [0,0,0,0,0]
            results_list['pred_leave'] = {'1':[],'2':[],'3':[],'4':[],'5':[]}
            if len(args.eval_cold):
                cold_results_list[eval_type]=[]
                cold_results_list['view_lengths'] = []
                hot_results_list[eval_type]=[]
                hot_results_list['view_lengths'] = []
        if args.save_logits:
            save_logits_gt_seq_for_evaluate=torch.empty((0, 82))
        print("Starts Inference")
        if args.count_view_completion:
            equal_num['test']=0
            equal_num['test_all']=0
        test_sample = 0
        for local_test_step, test_batch in tqdm.tqdm(enumerate(test_dataloader)):
            if args.debug: 
                if local_test_step>3:break # To test code
            test_batch = {k: v.cuda() for k, v in test_batch.items()}
            with torch.no_grad():
                # import pdb; pdb.set_trace()
                output = model(usr_image = test_batch["user_id"], usr_id = test_batch["user_id"], usr_mask = test_batch["user_mask"], 
                            vid_image = test_batch["photo_id"], vid_id = test_batch["photo_id"], vid_mask = test_batch["photo_mask"],
                            gt=test_batch["label"], mode="inference") 
            bsz = test_batch["user_id"].shape[0]
            test_sample+=bsz
            exposure_prob =  repeat(torch.tensor(args.exposure_prob).cuda(), "L -> b L", b=bsz)
            interests = torch.sigmoid(output["logits"])  * exposure_prob
            logits = output["logits"].detach().cpu()
            pred_label = torch.where(interests>args.threshold, 1.0, 0.0)
            gt = output["gt"]
            if args.count_view_completion:
                view_lengths = (test_batch["label"] == 1).sum(dim=1, keepdim=True)
                durations = (test_batch["label"] != -2).sum(dim=1, keepdim=True)
                equal_num['test'] += (view_lengths == durations).sum().item()
                equal_num['test_all']+=bsz

            if args.save_logits:
                combined = torch.cat((interests.cpu(), gt.cpu(), test_batch["user_id"].unsqueeze(-1).cpu(), test_batch["photo_id"].unsqueeze(-1).cpu()), dim=1)
                save_logits_gt_seq_for_evaluate = torch.cat((save_logits_gt_seq_for_evaluate, combined), dim=0)
            results_list = main_eval_batch(args, interests, gt, pred_label, results_list,type='inference', logits=logits)

            if len(args.eval_cold):
                cold_indices = [idx for idx, id_ in enumerate(test_batch["photo_id"].cpu().tolist()) if id_ not in train_videos_set]
                cold_count_inter+=len(cold_indices)
                cold_indices = torch.tensor(cold_indices, device=interests.device)
                cold_results_list = main_eval_batch(args, interests[cold_indices, :], gt[cold_indices, :], pred_label[cold_indices, :], cold_results_list, type='inference')
                hot_indices = [idx for idx, id_ in enumerate(test_batch["photo_id"].cpu().tolist()) if id_ in train_videos_set]
                hot_count_inter+=len(hot_indices)
                if len(hot_indices)>0:
                    hot_indices = torch.tensor(hot_indices, device=interests.device)
                    hot_results_list = main_eval_batch(args, interests[hot_indices, :], gt[hot_indices, :], pred_label[hot_indices, :], hot_results_list, type='inference')
        # import pdb; pdb.set_trace()
        if args.count_view_completion:
            print(equal_num)
            exit()

        final_result_dict = compute_final_result(results_list, test_sample)
        print(final_result_dict)
        logger.info(f"Test result, {final_result_dict}")

        if len(args.eval_cold):
            cold_final_result_dict = compute_final_result(cold_results_list)
            logger.info(f"Test result on cold videos, {cold_final_result_dict}")
            if args.count_view_completion:
                logger.info(f"{str(equal_num)}, cold_inter:{cold_count_inter}")
            hot_final_result_dict = compute_final_result(hot_results_list)
            logger.info(f"Test result on hot videos, {hot_final_result_dict}")
            if args.count_view_completion:
                logger.info(f"{str(equal_num)}, hot_inter:{hot_count_inter}")
            exit()
            
        if args.save_logits:
            name = f"{cur_time}_{args.num_layers_enc}_{args.exposure_prob_type}_{args.learning_rate}_{args.weight_decay}_{args.learnable_bias}_{args.loss_type}_{args.loss_weight}_{args.input_type}_{args.mask_loss}_{args.use_pe}_{args.fusion_heads}_earlystop_focal"
            torch.save(save_logits_gt_seq_for_evaluate, f'save_logits_gt_eval/result_{name}.pth')
        
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Example argument parser')
    parser = BaseReaderSeq_KuaiRand.parse_data_args(parser)

    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--valid_batch_size', type=int, default=1024)
    parser.add_argument('--test_batch_size', type=int, default=1024)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--learnable_bias', type=int, default=0)
    parser.add_argument('--wandb', type=int, default=0)
    parser.add_argument('--exp', type=str, default="")
    parser.add_argument('--logging_step', type=int, default=10)
    parser.add_argument('--valid_step', type=int, default=30)
    parser.add_argument('--ckpt_dir', type=str, default="ckpts_KuaiRand")
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--ff_dim', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=16)
    parser.add_argument('--num_query', type=int, default=1) # ?
    parser.add_argument('--num_clips', type=int, default=1) # ?
    parser.add_argument('--num_layers_enc', type=int, default=6) # change
    parser.add_argument('--num_layers_dec', type=int, default=0)
    parser.add_argument('--sr_ratio_lvls', type=list, default=[1,1,1,1,1]) 
    parser.add_argument('--use_patch_merge', type=list, default=[False, False, False, False, False]) 
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--iou_cutoff', type=float, default=0.7)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--topk_list', type=list, default=[])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--exposure_prob_type', type=str, default='ones', choices=['ones', 'statistics'])
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--input_type', type=dict, default={'user':'id','photo':'id'})
    parser.add_argument('--user_input_type', type=str, default='id')
    parser.add_argument('--photo_input_type', type=str, default='id')
    parser.add_argument('--loss_type', type=str, default='interestBPR', help='combination of focal, harzard, mse, huber, surviveCE, interestBPR, interestCE, interestKL, split by ,')
    parser.add_argument('--loss_type_list', type=list, default=[], help ='generate by code')
    parser.add_argument('--loss_weight', type=dict, default={'focal':1.0, 'mse':1.0, 'hazard':1.0, 'surviveCE':1.0, 'interestBPR':1.0, 'interestCE':1.0, 'interestKL':1.0})
    parser.add_argument('--loss_weight_surviveCE', type=float, default=1.0)
    parser.add_argument('--loss_weight_interestBPR', type=float, default=1.0)
    parser.add_argument('--loss_weight_interestCE', type=float, default=1.0)
    parser.add_argument('--use_pe', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=1)
    parser.add_argument('--save_logits', type=int, default=0)
    parser.add_argument('--eval_type_list', type=str, default='LeaveMSE,TOP1MSE,TOP_K', help='combination of JaccardSim, ProbAUC, LeaveMSE,LeaveCTR, LeaveCTR_view, TOP_K split by ,')
    parser.add_argument('--draw_case', type=int, default=0)
    parser.add_argument('--early_stop', type=int, default=20, help='The number of epochs when dev results drop continuously.')
    parser.add_argument('--main_metrics', type=str, default='HR@5')
    parser.add_argument('--TOP_K_permutation', type=int, default=1)
    parser.add_argument('--record_train_detail', type=int, default=0)
    parser.add_argument('--mask_loss', type=int, default=0)
    parser.add_argument('--count_view_completion', type=int, default=0)
    parser.add_argument('--TOP_K_mask', type=int, default=0)
    parser.add_argument('--fusion_heads', type=int, default=2)
    parser.add_argument('--eval_cold', type=str, default='', choices=['','test', 'validtest'])
    
    args = parser.parse_args() 
    exposure_prob = []
    if args.exposure_prob_type=="statistics":
        with open("data/KuaiRand_ExposureProb.json", "r") as f:
            probs = json.load(f)
            for idx in probs:
                exposure_prob.append(probs[idx])
    else:
        exposure_prob = [1.0] * 40
    args.exposure_prob = exposure_prob
    args.loss_weight["surviveCE"]=args.loss_weight_surviveCE
    args.loss_weight["interestBPR"]=args.loss_weight_interestBPR
    args.loss_weight["interestCE"]=args.loss_weight_interestCE
    if ',' in args.loss_type:
        args.loss_type_list = [item.strip() for item in args.loss_type.split(',')]
    else:
        args.loss_type_list = [args.loss_type.strip()]
    if ',' in args.eval_type_list:
        args.eval_type_list = [item.strip() for item in args.eval_type_list.split(',')]
    else:
        args.eval_type_list = [args.eval_type_list.strip()]
    
    if args.debug:
        args.epochs=2
        args.logging_step=1
        args.valid_step=1
        # args.train_batch_size = 128
        # args.valid_batch_size = 128
        # args.test_batch_size = 128
    if args.count_view_completion:
        args.epochs=1
        args.valid_step=200

    args.input_type['user'] = args.user_input_type
    args.input_type['photo'] = args.photo_input_type
    print(args)
    main(args)