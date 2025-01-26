# coding: utf-8
# @email: enoche.chow@gmail.com
"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from utils.metrics import metrics_dict
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time
import json


# These metrics are typical in topk recommendations
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        
        if 'SegMM' in config['dataset']:
            with open('data/photo_id2frame_id_leave_SegMM.json', 'r', encoding='utf-8') as f:
                self.photo_id2frame_id = json.load(f)
            with open('data/evaluate_test_User_Video_SegMM.json', 'r', encoding='utf-8') as f:
                self.evaluate_test_User_Video = json.load(f)
            with open('data/evaluate_dev_User_Video_SegMM.json', 'r', encoding='utf-8') as f:
                self.evaluate_valid_User_Video = json.load(f)
        with open('data/save_evaluate_all_User_Video_SegMM_coldTest.json', 'r', encoding='utf-8') as f:
            self.save_evaluate_all_User_Video_coldTest = json.load(f)
        if self.config.final_config_dict['save_logits']:
            if 'SegMM' in config['dataset']:
                with open('SegMM/second_map_user2id.json', 'r', encoding='utf-8') as f:
                    self.save_user2id = json.load(f)
                    self.save_id2user = {int(value):int(key) for key,value in self.save_user2id.items()}
                with open('data/save_evaluate_all_User_Video_SegMM.json', 'r', encoding='utf-8') as f:
                    self.save_evaluate_all_User_Video = json.load(f)

        self._check_args()

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def interest_TopK_mask(self, interests, view_lengths, durations):
        bsz, seq_len = interests.shape
        view_lengths = view_lengths.astype(np.int64)
        view_lengths = view_lengths.flatten()
        valid_row_mask = (view_lengths!=durations)
        bsz = np.sum(valid_row_mask)
        view_lengths = view_lengths[valid_row_mask]
        interests = interests[valid_row_mask, :]
        durations = durations[valid_row_mask]

        mask_batch = np.zeros((bsz, seq_len), dtype=bool)
        duration_broadcast = np.broadcast_to(durations[:, np.newaxis], (bsz, seq_len))
        mask_batch = np.arange(seq_len) < duration_broadcast

        interests = np.where(mask_batch, interests, float('inf'))

        permuted_indices = np.array([np.random.permutation(seq_len) for _ in range(bsz)])
        predictions = np.array([interests[i, permuted_indices[i]] for i in range(bsz)])  # 打乱数组

        sorted_indices = np.argsort(predictions, axis=1)

        # target_indices = permuted_indices[np.arange(bsz), view_lengths] # wrong!
        target_indices = np.argwhere(permuted_indices == view_lengths[:, np.newaxis])[:, 1]

        gt_rank = np.argmax(sorted_indices == target_indices[:, None], axis=1)+1

        evaluations={}
        for k in [1,3,5,10]:
            hit = (gt_rank <= k).astype(np.float32)
            for metric in ['hr','ndcg']:
                key = '{}@{}'.format(metric, k)
                if metric == 'hr':
                    evaluations[key] = hit.mean()
                elif metric == 'ndcg':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        print(evaluations)
        
        return evaluations

    def interest_TopK_nonmask(self, interests, view_lengths, durations):
        bsz, seq_len = interests.shape
        view_lengths = view_lengths.astype(np.int64)
        view_lengths = view_lengths.flatten()
        valid_row_mask = (view_lengths<40)
        bsz = np.sum(valid_row_mask)
        view_lengths = view_lengths[valid_row_mask]
        interests = interests[valid_row_mask, :]

        permuted_indices = np.array([np.random.permutation(seq_len) for _ in range(bsz)])
        predictions = np.array([interests[i, permuted_indices[i]] for i in range(bsz)])  # 打乱数组

        sorted_indices = np.argsort(predictions, axis=1)

        # target_indices = permuted_indices[np.arange(bsz), view_lengths] # wrong!
        target_indices = np.argwhere(permuted_indices == view_lengths[:, np.newaxis])[:, 1]

        gt_rank = np.argmax(sorted_indices == target_indices[:, None], axis=1)+1

        evaluations={}
        for k in [1,3,5,10]:
            hit = (gt_rank <= k).astype(np.float32)
            for metric in ['hr','ndcg']:
                key = '{}@{}'.format(metric, k)
                if metric == 'hr':
                    evaluations[key] = hit.mean()
                elif metric == 'ndcg':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        print(evaluations) # need to debug !!!
        
        return evaluations

    def save_logits(self,scores_dict, mode):
        print('to get logtis for saving')
        logits_dict={}
        for userID in self.save_evaluate_all_User_Video:
            for photo_id in self.save_evaluate_all_User_Video[userID]:
                frame_ids = self.photo_id2frame_id[str(photo_id)]
                time = self.save_evaluate_all_User_Video[userID][photo_id]['time']
                user_id = self.save_id2user[int(userID)]
                
                if mode=='0':
                    scores_tmp = [float(scores_dict[int(userID)][frameid]) for frameid in frame_ids] + [float(0)]*(40-len(frame_ids))
                elif mode=='default':
                    last = scores_dict[int(userID)][-1]
                    scores_tmp = [float(scores_dict[int(userID)][frameid]) for frameid in frame_ids] + [float(last)]*(40-len(frame_ids))
                elif mode=='fill':
                    scores_tmp = [float(scores_dict[int(userID)][frameid]) for frameid in frame_ids]
                logits_dict[f"{user_id}-{photo_id}-{time}"] = scores_tmp
        

        print('start save', len(logits_dict))
        config_dict = self.config.final_config_dict
        save_path = f"{config_dict['dataset']}-{config_dict['model']}._logits.pth"
        with open(save_path.replace(".pth", ".json"), "w") as fw:
            json.dump(logits_dict, fw)
        torch.save(logits_dict, save_path)
        print('save done', save_path)
        # exit() # TODO

    def evaluate(self, batch_matrix_list, eval_data, scores_dict, logger, is_test=False, idx=0, update_flag=False):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
                model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)
        # if recom right?
        if is_test:
            evaluate_User_Video = self.evaluate_test_User_Video
            print('is_test in topk_evaluator.evaluate')

        else:
            evaluate_User_Video = self.evaluate_valid_User_Video
            print('Is valid in topk_evaluator.evaluate')
        
        
        scores_s=[]
        view_lengths=[]
        durations=[]
        if self.config['dataset'] == 'KwaiMMleave': # mask
            for userID in evaluate_User_Video:
                for photo_id in evaluate_User_Video[userID]:
                    frame_ids = self.photo_id2frame_id[str(photo_id)]
                    scores_tmp = [scores_dict[int(userID)][frameid] for frameid in frame_ids] + [0]*(40-len(frame_ids))
                    scores_s.append(scores_tmp)
                    view_lengths.append(evaluate_User_Video[userID][photo_id]['view_length'])
                    durations.append(evaluate_User_Video[userID][photo_id]['duration'])
            if is_test and update_flag and self.config.final_config_dict['save_logits']:
                self.save_logits(scores_dict, mode='0')
            metric_dict = self.interest_TopK_mask(np.array(scores_s), np.array(view_lengths), np.array(durations))
        elif self.config['dataset'] == 'KwaiMMleaveDefault' or self.config['dataset'] == 'SegMMdefault': # non-mask, filled by last
            coldIndex=[]
            now_index=0
            for userID in evaluate_User_Video:
                for photo_id in evaluate_User_Video[userID]:
                    if userID in self.save_evaluate_all_User_Video_coldTest:
                        if photo_id in self.save_evaluate_all_User_Video_coldTest[userID]:
                            coldIndex.append(now_index)
                    frame_ids = self.photo_id2frame_id[str(photo_id)]
                    last = scores_dict[int(userID)][-1]
                    scores_tmp = [scores_dict[int(userID)][frameid] for frameid in frame_ids] + [last]*(40-len(frame_ids))
                    scores_s.append(scores_tmp)
                    view_lengths.append(evaluate_User_Video[userID][photo_id]['view_length'])
                    durations.append(evaluate_User_Video[userID][photo_id]['duration'])
                    now_index+=1
            if is_test and update_flag and self.config.final_config_dict['save_logits']:
                self.save_logits(scores_dict, mode='default')
            if is_test and update_flag and self.config.final_config_dict['test_cold']:
                cold_metrics_dict = self.interest_TopK_mask(np.array(scores_s)[coldIndex], np.array(view_lengths)[coldIndex], np.array(durations)[coldIndex])
                Allindices = np.arange(np.array(scores_s).shape[0])
                hotIndex = np.isin(Allindices, coldIndex, invert=True)
                hot_metrics_dict = self.interest_TopK_mask(np.array(scores_s)[hotIndex], np.array(view_lengths)[hotIndex], np.array(durations)[hotIndex])
                # import pdb; pdb.set_trace()
                logger.info('cold result:'+str(cold_metrics_dict))
                logger.info('hot result:'+str(hot_metrics_dict))
                
            metric_dict = self.interest_TopK_nonmask(np.array(scores_s), np.array(view_lengths), np.array(durations))
            
        elif self.config['dataset'] == 'KwaiMMleaveFill': # non-mask
            for userID in evaluate_User_Video:
                for photo_id in evaluate_User_Video[userID]:
                    frame_ids = self.photo_id2frame_id[str(photo_id)]
                    last = scores_dict[int(userID)][-1]
                    scores_tmp = [scores_dict[int(userID)][frameid] for frameid in frame_ids]
                    scores_s.append(scores_tmp)
                    view_lengths.append(evaluate_User_Video[userID][photo_id]['view_length'])
                    durations.append(evaluate_User_Video[userID][photo_id]['duration'])
            if is_test and update_flag and self.config.final_config_dict['save_logits']:
                self.save_logits(scores_dict, mode='fill')
            metric_dict = self.interest_TopK_nonmask(np.array(scores_s), np.array(view_lengths), np.array(durations))

        else:
            bool_rec_matrix = []
            for m, n in zip(pos_items, topk_index):
                bool_rec_matrix.append([True if i in m else False for i in n])
            bool_rec_matrix = np.asarray(bool_rec_matrix)

            # get metrics
            metric_dict = {}
            result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
            for metric, value in zip(self.metrics, result_list):
                for k in self.topk:
                    key = '{}@{}'.format(metric, k)
                    metric_dict[key] = round(value[k - 1], 4)
        return metric_dict

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in topk_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [topk_metrics[metric.lower()] for metric in self.metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
