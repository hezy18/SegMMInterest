
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors

import torch
import os
import random

seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  

# Jaccard Similarity :
def IoU_binary(pred, label, view_length, duration, type='length_aware'):
    I = (pred == label)
    U = 2 * (view_length+1)
    I_length_aware = I[:view_length+1]
    U_length_aware = 2 * (duration)
    IoU_original = I / U
    IoU_length_aware = I_length_aware / U_length_aware
    if type=='original':
        return IoU_original
    elif type=='length_aware':
        return IoU_length_aware
    else:
        raise ValueError(
            f"Invalid Value for IoU type: Supported 'original', 'length_aware'"
        )


def IoU_Sim(logit, label, view_length, duration, type='length_aware'):
    diff = abs(label-logit)
    I = (1-diff).cpu().tolist()
    I_original = I[:view_length]
    I_length_aware = I_original + [1.0] * (duration - view_length)
    U_original = view_length
    U_length_aware = duration
    assert U_original == len(I_original)
    assert U_length_aware == len(I_length_aware)
    IoU_original = sum(I_original) / U_original
    IoU_length_aware = sum(I_length_aware) / U_length_aware

    if type=='original':
        return IoU_original.item()
    elif type=='length_aware':
        return IoU_length_aware.item()
    else:
        raise ValueError(
            f"Invalid Value for IoU type: Supported 'original', 'length_aware'"
        )

# AUC
def ProbAUC(prob, label, mask): # logit (1, 40)
    masked_prob = prob[mask == 1]
    masked_label = label[mask == 1]
    masked_label = torch.where(masked_label == -1, torch.tensor(0), masked_label)
    # import pdb; pdb.set_trace()
    valid_prob = masked_prob.detach().cpu().numpy()
    valid_label = masked_label.detach().cpu().numpy()
    # import pdb; pdb.set_trace()

    auc = roc_auc_score(valid_label, valid_prob)
    return auc # 如果label都是0，则一定报错？(



def ProbAUC_batch(probs, labels, masks): # logit (1, 40)
    masked_probs = probs[masks == 1]
    masked_labels = labels[masks == 1]
    masked_labels = torch.where(masked_labels == -1, torch.tensor(0), masked_labels)
    valid_probs = masked_probs.detach().cpu().numpy().flatten()
    valid_labels = masked_labels.detach().cpu().numpy().flatten()
    auc = roc_auc_score(valid_labels, valid_probs)
    return auc

def predict_view_length(prob, mask):
    masked_prob = prob[mask == 1]
    pred_view_length = torch.sum(masked_prob).item()
    return pred_view_length

def LeaveCTR(interest, survival_prob, view_length):
    CTR = 1-interest[view_length-1].item()
    CTR2 = 1-survival_prob[view_length-1].item()
    return CTR, CTR2

def TOP_K_leave_mask_scaled(interests, view_lengths, mask_batch, permutation=1): 
    bsz, seq_len = interests.shape

    view_lengths = view_lengths.astype(np.int64)
    view_lengths = view_lengths.flatten()
    valid_row_mask = (view_lengths!=mask_batch.sum(axis=1))
    bsz = np.sum(valid_row_mask)
    view_lengths = view_lengths[valid_row_mask]
    interests = interests[valid_row_mask, :]
    mask_batch = mask_batch[valid_row_mask, :]

    interests = np.where(mask_batch, interests, 1.1)

    durations = mask_batch.sum(axis=1) #
    if permutation:

        permuted_indices = np.array([np.random.permutation(seq_len) for _ in range(bsz)])
        predictions = np.array([interests[i, permuted_indices[i]] for i in range(bsz)])  

        sorted_indices = np.argsort(predictions, axis=1)

        # target_indices = permuted_indices[np.arange(bsz), view_lengths] # wrong!
        target_indices = np.argwhere(permuted_indices == view_lengths[:, np.newaxis])[:, 1]

        gt_rank = np.argmax(sorted_indices == target_indices[:, None], axis=1)+1
    else:
        predictions = interests
        sorted_indices = np.argsort(predictions, axis=1)
        gt_rank = np.argmax(sorted_indices == view_lengths[:, None], axis=1)+1
    gt_rank = (gt_rank-1)*40/durations+1

    evaluations={}
    for k in [1,3,5,10]:
        hit = (gt_rank <= k).astype(np.float32)
        for metric in ['HR','NDCG']:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    print(evaluations)
    return evaluations

def TOP_K_leave_mask(interests, view_lengths, mask_batch, permutation=1): 
    bsz, seq_len = interests.shape

    view_lengths = view_lengths.astype(np.int64)
    view_lengths = view_lengths.flatten()
    valid_row_mask = (view_lengths!=mask_batch.sum(axis=1))
    bsz = np.sum(valid_row_mask)
    view_lengths = view_lengths[valid_row_mask]
    interests = interests[valid_row_mask, :]
    mask_batch = mask_batch[valid_row_mask, :]

    interests = np.where(mask_batch, interests, 1.1)

    if permutation:

        permuted_indices = np.array([np.random.permutation(seq_len) for _ in range(bsz)])
        predictions = np.array([interests[i, permuted_indices[i]] for i in range(bsz)]) 

        sorted_indices = np.argsort(predictions, axis=1)

        # target_indices = permuted_indices[np.arange(bsz), view_lengths] # wrong!
        target_indices = np.argwhere(permuted_indices == view_lengths[:, np.newaxis])[:, 1]

        gt_rank = np.argmax(sorted_indices == target_indices[:, None], axis=1)+1
    else:
        predictions = interests
        sorted_indices = np.argsort(predictions, axis=1)
        gt_rank = np.argmax(sorted_indices == view_lengths[:, None], axis=1)+1

    evaluations={}
    for k in [1,3,5,10]:
        hit = (gt_rank <= k).astype(np.float32)
        for metric in ['HR','NDCG']:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    print(evaluations)
    return evaluations

def TOP_K_leave(interests, view_lengths, mask_batch, permutation=1, test=0): # numpy array
    bsz, seq_len = interests.shape

    min_indices = np.argmin(interests, axis=1) # for MSE/MAE/xAUC

    view_lengths = view_lengths.astype(np.int64)
    view_lengths = view_lengths.flatten()
    valid_row_mask = (view_lengths<40) 
    bsz = np.sum(valid_row_mask)
    view_lengths = view_lengths[valid_row_mask]
    interests = interests[valid_row_mask, :]
    mask_batch = mask_batch[valid_row_mask, :]

    if permutation:

        permuted_indices = np.array([np.random.permutation(seq_len) for _ in range(bsz)])
        predictions = np.array([interests[i, permuted_indices[i]] for i in range(bsz)])  

        sorted_indices = np.argsort(predictions, axis=1)

        # target_indices = permuted_indices[np.arange(bsz), view_lengths]# wrong!
        target_indices = np.argwhere(permuted_indices == view_lengths[:, np.newaxis])[:, 1]

        gt_rank = np.argmax(sorted_indices == target_indices[:, None], axis=1)+1
        # import pdb; pdb.set_trace()
    else:
        predictions = interests
        sorted_indices = np.argsort(predictions, axis=1)
        # index_positions = np.where(sorted_indices == view_lengths)
        gt_rank = np.argmax(sorted_indices == view_lengths[:, None], axis=1)+1
    
    evaluations={}
    for k in [1,3,5,10]:

        # hit = np.any(sorted_indices[:, :k] == view_lengths[:, None], axis=1)
        hit = (gt_rank <= k).astype(np.float32)
        # if equal.sum()>40:
        #     indices = gt_rank == 1
        #     hit[indices] = k / equal[indices]
        for metric in ['HR','NDCG']:
            key = '{}@{}'.format(metric, k)
            if metric == 'HR':
                evaluations[key] = hit.mean()
            elif metric == 'NDCG':
                evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
            else:
                raise ValueError('Undefined evaluation metric: {}.'.format(metric))
    print(evaluations)
    if test:
        return evaluations, min_indices
    else:
        return evaluations

def draw_hotmap(x1, gt1, uid_pid, ckpt_path, dir_path='case'):

    print(x1)
    print(gt1)

    # import pdb; pdb.set_trace()

    cmap_data = [
    (0.0, mcolors.to_rgba('white')),
    (0.5, mcolors.to_rgba('red')),
    (1, mcolors.to_rgba('red'))   
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_hot', cmap_data)

    print(x1.shape, gt1.shape)
    hot_map_data = np.stack((x1, gt1), axis=0)
    print(hot_map_data.shape)
    plt.figure(figsize=(8, 4))
    for j in range(2):
        plt.subplot(2, 1, j+1)  
        plt.imshow(hot_map_data[j].reshape(1, -1), cmap=custom_cmap, norm=Normalize(vmin=0, vmax=1))
        plt.title(['interest', 'leavegt'][j])

        for k in range(len(hot_map_data[j])):
            plt.text(k, 0, f'{hot_map_data[j][k]:.6f}', ha='center', va='center', color='black',fontsize=5)

    plt.suptitle(f'{uid_pid}')
    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9, hspace=0.5, wspace=0.5)
    path = ckpt_path.split('/')[-2]+'_'+ckpt_path.split('/')[-1].split('.pth')[0]
    plt.savefig(f'figure/{dir_path}/{path}_{uid_pid}.png')

def main_eval_batch(args, interests, ground_truths, pred_labels, results_list, type='inference', test_type='new', logits=None):
    # import pdb; pdb.set_trace()
    mask_batch = ground_truths!=-2
    batch_size = interests.shape[0]

    
    if test_type == 'old':
        survival_probs = interests
    else:
        h_t = torch.cumsum(torch.log(interests), dim=1)
        survival_probs = torch.exp(h_t)
    view_lengths = (ground_truths == 1).sum(dim=1, keepdim=True).cpu().numpy()
    # view_lengths_torch = (ground_truths==1).sum(dim=1, keepdim=True).to(interests.device)
    durations = (ground_truths != -2).sum(dim=1, keepdim=True).cpu().numpy()

    # survival_probs_masked = survival_probs.clone()
    # survival_probs_masked[~mask] = 0
    # pred_view_lengths = torch.sum(survival_probs_masked, dim=1)
    if 'ProbAUC' in results_list:
        auc_batch = ProbAUC_batch(survival_probs, ground_truths, mask_batch)
        results_list['ProbAUC'].append(float(auc_batch))
    i=0

    if 'TOP_K' in results_list:
        # import pdb; pdb.set_trace()
        if args.TOP_K_mask:
            
            # evaluations = TOP_K_leave_mask_scaled(interests.cpu().detach().numpy(), view_lengths, mask_batch.cpu().detach().numpy(), permutation = args.TOP_K_permutation)
            evaluations = TOP_K_leave_mask(interests.cpu().detach().numpy(), view_lengths, mask_batch.cpu().detach().numpy(), permutation = args.TOP_K_permutation)
        else:
            if 'TOP1MSE' in results_list:
                evaluations, Top1pos = TOP_K_leave(interests.cpu().detach().numpy(), view_lengths, mask_batch.cpu().detach().numpy(), permutation = args.TOP_K_permutation, test=1)
                results_list['TOP1MSE'].append(Top1pos)
            else:
                evaluations = TOP_K_leave(interests.cpu().detach().numpy(), view_lengths, mask_batch.cpu().detach().numpy(), permutation = args.TOP_K_permutation)
        
        for metrics in evaluations:
            if metrics not in results_list:
                results_list[metrics]=[]
            results_list[metrics].append(float(evaluations[metrics]))
    # print(view_lengths)
    # print(durations)
    # import pdb;pdb.set_trace()
    if logits!=None:
        softmax_logits = torch.nn.functional.softmax(logits, dim=1)
        inv_sm_logits = 1/softmax_logits
        leave_p1 = inv_sm_logits / inv_sm_logits.sum(dim=1).unsqueeze(1)
        pos = torch.linspace(0, 39, 40)
        pred_leave1 = torch.sum(leave_p1 * pos, dim=1)

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mae1 = mean_absolute_error(view_lengths, np.array(pred_leave1.int()))
        results_list['MAES']+=mae1*batch_size
        print(results_list['MAES'])
        results_list['pred_leave'].append(pred_leave1.int())

    for pred, interest, survival_prob, label, mask in zip(pred_labels, interests, survival_probs, ground_truths, mask_batch):
        view_length = (label==1).sum().cpu()
        duration = (label!=-2).sum().cpu()

        if args.draw_case:
            if duration>20:
                draw_interest = interest[:duration].cpu().numpy()
                draw_view_prob = survival_prob[:duration].cpu().numpy()
                draw_gold = label[:duration].cpu().numpy()
                # draw_gold = [0 if x == -1 else x for x in draw_gold]
                draw_hotmap(draw_interest,draw_view_prob,draw_gold,i) 
                i+=1


        for eval_type in results_list:
            if eval_type == 'JaccardSim':
                IoU = IoU_Sim(survival_prob, label, view_length, duration, type='length_aware')
                results_list[eval_type].append(IoU)
            # elif eval_type == 'ProbAUC':
            #     auc = ProbAUC(survival_prob, label, mask)
            #     results_list[eval_type].append(auc)
            elif eval_type == 'LeaveMSE':
                pred_y = predict_view_length(survival_prob, mask)
                results_list[eval_type].append(float(pred_y))
                results_list['view_lengths'].append(float(view_length.item()))
                if 'duration_lengths' in results_list:
                    results_list['duration_lengths'].append(float(duration.item()))
                # import pdb; pdb.set_trace()
            elif eval_type == 'LeaveCTR':
                CTR, _ = LeaveCTR(interest, survival_prob, view_length)
                results_list[eval_type].append(float(CTR))
            elif eval_type == 'LeaveCTR_view':
                _, CTR = LeaveCTR(interest, survival_prob, view_length)
                results_list[eval_type].append(float(CTR))
            
            
            
    return results_list

