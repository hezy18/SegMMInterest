
""" Clip based recommendation （WideDeep） - Zhiyu He
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from utils.layers import MLP_Block
import torch.nn.functional as F

class ClipRecBase(object):
    @staticmethod
    def parse_model_args_Clip(parser):
        parser.add_argument('--emb_dim', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--contrastive', type=str, default='',
                            help='Choose the contrastive loss if used')
        parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer in the MLP module.")
        parser.add_argument('--adjust_interest_weight', type=int, default=0,
							help="Whether to adjust interest weight or not.")
        parser.add_argument('--duration_mask', type=int, default=0,
                            help="Whether to get duration mask or not.")
        return parser
    
    def _define_init(self, args, corpus):
        self.embedding_dim = args.emb_dim
        
        self.dnn_layers = eval(args.dnn_layers)
        self.dropout = args.dropout

        self.contrastive = args.contrastive
        self.adjust_interest_weight = args.adjust_interest_weight
        self.duration_mask = args.duration_mask

        self._define_params_ClipRec()
        self.apply(self.init_weights)


    def _define_params_ClipRec(self):
        
        # self.embedding_dict = nn.ModuleDict()
        # for f in self.context_features:
        #     self.embedding_dict[f] = nn.Embedding(self.feature_max[f],self.embedding_dim) if f.endswith('_c') or f.endswith('_id') else\
		# 			nn.Linear(1,self.embedding_dim,bias=False)
        # self.feature_dim = self.embedding_dim * len(self.embedding_dict)
        
        # fm
        self.user_embedding = nn.Embedding(self.feature_max['user_id'], self.embedding_dim)
        self.item_embedding = nn.Embedding(self.feature_max['item_id'], self.embedding_dim)
        self.frame_position_embedding = nn.Linear(1, self.embedding_dim)
        self.frame_feature_dim=1024
        self.frame_embedding = nn.Linear(self.frame_feature_dim, self.embedding_dim)
        self.frame_id_projector = nn.Linear(self.embedding_dim*2, self.embedding_dim)
        
        
        # linear
        self.user_linear = nn.Embedding(self.feature_max['user_id'], 1)
        self.item_linear = nn.Embedding(self.feature_max['item_id'], 1)
        self.frame_position_linear = nn.Linear(1, 1)
        self.frame_linear = nn.Linear(self.frame_feature_dim, 1)
        self.frame_id_projector_linear = nn.Linear(2, 1)
        
        self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

        if self.adjust_interest_weight:
            self.trainable_interest_weight = torch.nn.Parameter(torch.ones(40), requires_grad=True) # 这里是对所有交互固定的clip概率，而不是f(user item time)

        pre_size = 3 * self.embedding_dim
        self.dnn_mlp_layers = MLP_Block(pre_size, self.dnn_layers, hidden_activations="ReLU",
							   batch_norm=False, dropout_rates=self.dropout, output_dim=1)
	


    def _get_embeddings_ClipRec(self, feed_dict):
        user_ids = feed_dict['user_id'] # batch_size
        item_ids = feed_dict['item_id'] # batch_size * item_num (e.g., 2 for training)
        # print(feed_dict.keys())
        # print('user_ids',user_ids.shape)
        # print('item_ids',item_ids.shape)
        
        ### fm
        # frames_feats
        if 'i_item_frames' in feed_dict:
            item_frames = feed_dict['i_item_frames'] # batch_size * item_num * clip_num(padding) * feature_dim(1024)
            batch_size, item_num, clip_num, _ = item_frames.shape
            # frame feats
            frame_feats_embed = torch.relu(self.frame_embedding(item_frames)) # batch_size * item_num * clip_num * embedding_dim
            frame_feats_value =torch.relu(self.frame_linear(item_frames))
        else:
            batch_size, item_num = item_ids.shape
            clip_num=40
        
        # frame position
        frame_positions = torch.arange(clip_num).unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1).float()   # batch_size * item_num * clip_num
        # print('frame_positions',frame_positions.shape, frame_positions.dtype)
        
        # ids embedding
        user_embed = self.user_embedding(user_ids)  # batch_size * embedding_dim
        
        item_embed = self.item_embedding(item_ids)  # batch_size * item_num * embedding_dim
        item_embed_expanded = item_embed.unsqueeze(2).expand(-1, -1, clip_num, -1)  # batch_size * item_num * clip_num * embedding_dim
        # print('item_embed_expanded',item_embed_expanded.shape)
        
        frame_positions = frame_positions.unsqueeze(-1).to(self.device)
        # print('frame_positions',frame_positions.shape)
        frame_position_embed = self.frame_position_embedding(frame_positions)  # batch_size * item_num * clip_num * embedding_dim
        # print('frame_position_embed',frame_position_embed.shape)
        
        # item_frames = item_frames.view(-1, feature_dim)  # (batch_size * item_num * clip_num) * feature_dim
        # frame_feats_embed = torch.relu(self.frame_embedding(item_frames))  # (batch_size * item_num * clip_num) * embedding_dim
        # frame_feats_embed = frame_feats_embed.view(batch_size, item_num, clip_num, -1)  # batch_size * item_num * clip_num * embedding_dim
        
        ### linear
        user_value = self.user_linear(user_ids)  # batch_size * 1
        item_value = self.item_linear(item_ids)  # batch_size * item_num * 1
        item_value_expanded = item_value.unsqueeze(2).expand(-1, -1, clip_num, -1)  # batch_size * item_num * clip_num * 1
        frame_position_linear = self.frame_position_linear(frame_positions)  # batch_size * item_num * clip_num * 1
        
        
        if 'i_item_frames' in feed_dict:
            frame_id_embed = self.frame_id_projector(torch.cat([item_embed_expanded, frame_position_embed], dim=-1)) # batch_size * item_num * clip_num * embedding_dim
            frame_id_value = self.frame_id_projector_linear(torch.cat([item_value_expanded, frame_position_linear], dim=-1)) # batch_size * item_num * clip_num * 2
            return user_embed, user_value, frame_feats_embed, frame_feats_value, frame_id_embed, frame_id_value
        else:
            frame_id_embed = torch.cat([item_embed_expanded, frame_position_embed], dim=-1) # batch_size * item_num * clip_num * (2*embedding_dim)
            frame_id_value = torch.cat([item_value_expanded, frame_position_linear], dim=-1) # batch_size * item_num * clip_num * 2
            return user_embed, user_value, frame_id_embed, frame_id_value


    def forward(self, feed_dict):
        
        if 'i_item_frames' in feed_dict:
            user_embed, user_value, frame_feats_embed, frame_feats_value, frame_id_embed, frame_id_value = self._get_embeddings_ClipRec(feed_dict)
            frame_concat_embed = torch.cat([frame_feats_embed, frame_id_embed], dim=-1)  
            frame_concat_value = torch.cat([frame_feats_value,frame_id_value], dim=-1) 
        
        else:
            user_embed, user_value, frame_id_embed, frame_id_value = self._get_embeddings_ClipRec(feed_dict)
            frame_concat_embed = frame_id_embed
            frame_concat_value = frame_id_value
        # print('frame_concat_embed',frame_concat_embed.shape)
        batch_size, item_num, clip_num, _ = frame_concat_embed.shape
        user_embed = user_embed.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
        # print('user_embed',user_embed.shape)
        fm_vectors = torch.cat([user_embed, frame_concat_embed], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)
        # print('fm_vectors',fm_vectors.shape)
        deep_prediction = self.dnn_mlp_layers(fm_vectors).squeeze(dim=-1)
        # print('deep_prediction',deep_prediction.shape)
        
        user_value = user_value.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
        linear_value = torch.cat([user_value, frame_concat_value], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)
        # print('linear_value',linear_value.shape)
        wide_prediction = self.overall_bias + linear_value.sum(dim=-1)
        # print('wide_prediction',wide_prediction.shape)
        
        clip_predictions = deep_prediction + wide_prediction
        
        # weight
        if self.adjust_interest_weight:
            interest_weight = self.trainable_interest_weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1) # batch_size*item_num*clip_num
        else:
            if 'c_interest_weight' in feed_dict:
                interest_weight = feed_dict['c_interest_weight'] # batch_size * item_num * clip_num(padding)
            else:
                interest_weight = torch.ones((batch_size, item_num, clip_num), device=frame_concat_embed.device)

        # mask
        item_duration = feed_dict['i_duration'] # batch_size * item_num; 代表每个视频有多少clip
        if self.duration_mask:
            mask = torch.arange(clip_num).unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1).to(item_duration.device) < item_duration.unsqueeze(-1)  # batch_size * item_num * clip_num
            mask = mask.float()  # 转换为浮点型
        else:
            mask = torch.ones((batch_size, item_num, clip_num), device=item_duration.device)
        
        weighted_predictions = clip_predictions * interest_weight * mask
        
        predictions = weighted_predictions.sum(dim=-1)
        
        if self.contrastive == 'ContrastiveLoss': # frame_feats_embed v.s. frame_id_embed
            # import pdb; pdb.set_trace()
            contrastive_loss = ContrastiveLoss()
            labels = torch.ones(batch_size * item_num * clip_num).to(frame_feats_embed.device)  # 用于对比学习的标签，这里只进行了相同项目的对比学习。TODO：需要构造0 label的对比 
            loss_contrastive = contrastive_loss(frame_feats_embed.view(-1, self.embedding_dim), frame_id_embed.view(-1, self.embedding_dim), labels)
            return {'prediction':predictions, 'contrastive_loss': loss_contrastive}
        elif self.contrastive == 'infoNCELoss':
            contrastive_loss = infoNCELoss()
            # item-level contrastive
            e = torch.cat((frame_feats_embed.view(batch_size*item_num, clip_num*self.embedding_dim), frame_feats_value.view(batch_size*item_num, clip_num*1)), dim=1)
            g = torch.cat((frame_id_embed.view(batch_size*item_num, clip_num*self.embedding_dim), frame_id_value.view(batch_size*item_num, clip_num*1)), dim=1)
            # clip-level contrastive:TODO
            loss_contrastive = contrastive_loss(e,g)
            return {'prediction':predictions, 'contrastive_loss': loss_contrastive}
        else:
            return {'prediction':predictions}


class ClipRecCTR(ContextCTRModel, ClipRecBase):
    reader, runner = 'ContextReader', 'CTRRunner'
    extra_log_args = ['emb_dim', 'dnn_layers', 'contrastive', 'loss_n','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']

    @staticmethod
    def parse_model_args(parser):
        parser = ClipRecBase.parse_model_args_Clip(parser)
        return ContextCTRModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextCTRModel.__init__(self, args, corpus)
        self._define_init(args,corpus)

    def forward(self, feed_dict):
        out_dict = ClipRecBase.forward(self, feed_dict)
        out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
        out_dict['label'] = feed_dict['label'].view(-1) 
        return out_dict

class ClipRecRanking(ContextModel, ClipRecBase):
    reader, runner = 'ContextReader', 'BaseRunner'
    extra_log_args = ['emb_dim', 'dnn_layers', 'contrastive','loss_n','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']

    @staticmethod
    def parse_model_args(parser):
        parser = ClipRecBase.parse_model_args_Clip(parser)
        return ContextModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        ContextModel.__init__(self, args, corpus)
        self._define_init(args,corpus)

    def forward(self, feed_dict):
        return ClipRecBase.forward(self, feed_dict)
    

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, embedding1, embedding2, label):
        distance = (embedding1 - embedding2).pow(2).sum(1)  # 欧氏距离
        loss = 0.5 * (label.float() * distance + (1 - label.float()) * torch.relu(self.margin - torch.sqrt(distance + 1e-6)).pow(2))
        return loss.mean()

class infoNCELoss(nn.Module): # For CTR
    def __init__(self, tau=0.1):
        super(infoNCELoss, self).__init__()
        self.tau = tau
    
    def forward(self, e, g): 
        '''
        e,g size: item_num, feature_dim
        '''
        item_num, _ = e.shape

        e = F.normalize(e, dim=-1) 
        g = F.normalize(g, dim=-1)

        dot_products = torch.mm(e, g.transpose(0, 1)) / self.tau  # batch_size, item_num, item_num
        # dot_products = torch.sum(torch.sigmoid(e)*torch.sigmoid(g), dim=1)
        mask = torch.eye(item_num, device=e.device)
        positive_scores = torch.exp(dot_products) * mask
        negative_scores = torch.exp(dot_products) * (1 - mask)
        
        pos_sum = positive_scores.sum(dim=1)
        neg_sum = negative_scores.sum(dim=1)
        
        loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        return loss.mean()