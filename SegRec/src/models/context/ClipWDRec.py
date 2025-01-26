
""" WideDeep
Reference:
  Wide {\&} Deep Learning for Recommender Systems, Cheng et al. 2016. The 1st workshop on deep learning for recommender systems.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextModel, ContextCTRModel
from utils.layers import MLP_Block
import torch.nn.functional as F

class ClipWDRecBase(object):
	@staticmethod
	def parse_model_args_ClipWD(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.")
		parser.add_argument('--contrastive', type=str, default='', help='Choose the contrastive loss if used')
		parser.add_argument('--adjust_interest_weight', type=int, default=0, 
							help="Whether to adjust interest weight or not.")
		parser.add_argument('--train_module', type=int, default=0, help="Whether to train module for interest_weight or not.")
		parser.add_argument('--duration_mask', type=int, default=0, help="Whether to get duration mask or not.")
		parser.add_argument('--norm_interest_type', type=str, default='none', choices=['softmax', 'sigmoid','none'], 
							help="Which normalized type to use on interest logits.") # original:softmax_interest,but now Do on logit
		return parser

	def _define_init(self, args, corpus):
		self.layers = eval(args.layers)
		self.vec_size = args.emb_size
		self.contrastive = args.contrastive
		self.adjust_interest_weight = args.adjust_interest_weight
		self.duration_mask = args.duration_mask
		self.norm_interest_type = args.norm_interest_type

		self._define_params_ClipWD()
		self.apply(self.init_weights)

	def _define_params_ClipWD(self):

        # fm
		self.user_embedding = nn.Embedding(self.feature_max['user_id'], self.vec_size)
		self.item_embedding = nn.Embedding(self.feature_max['item_id'], self.vec_size)
		self.frame_position_embedding = nn.Linear(1, self.vec_size)
		if len(self.clip_feature_path):
			self.frame_feature_dim=1024
			self.frame_embedding = nn.Linear(self.frame_feature_dim, self.vec_size)
			self.frame_id_projector = nn.Linear(self.vec_size*2, self.vec_size)
        
        # linear
		self.user_linear = nn.Embedding(self.feature_max['user_id'], 1)
		self.item_linear = nn.Embedding(self.feature_max['item_id'], 1)
		self.frame_position_linear = nn.Linear(1, 1)
		if len(self.clip_feature_path):
			self.frame_linear = nn.Linear(self.frame_feature_dim, 1)
			self.frame_id_projector_linear = nn.Linear(2, 1)

		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

		if self.adjust_interest_weight:
			self.trainable_interest_weight = torch.nn.Parameter(torch.ones(40), requires_grad=True) # 这里是对所有交互固定的clip概率，而不是f(user item time)
		
		# if len(self.clip_feature_path):
		# 	self.item_frames_context_embedding = nn.Linear(1024, self.vec_size, bias=False) # reader 占有存储空间的策略
		# 	self.item_frames_linear_embedding = nn.Linear(1024, 1, bias=False)
		# deep layers
		# if len(self.clip_feature_path):
		# 	self.item_frame_number = 40
		# else:
		# 	self.item_frame_number = 0
		if hasattr(self, 'layers'):
			pre_size = 3 * self.vec_size
			self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
									batch_norm=False, dropout_rates=self.dropout, output_dim=1)
	

	def _get_embeddings_ClipWDRec(self, feed_dict):
		## 1. item embedding
		item_ids = feed_dict['item_id'] # batch_size * item_num
		if 'i_item_frames' in feed_dict:
			item_frames = feed_dict['i_item_frames'] # batch_size * _get_embeddings_ClipRec * clip_num(padding) * feature_dim(1024)
			batch_size, item_num, clip_num, _ = item_frames.shape
			frame_feats_embed = torch.relu(self.frame_embedding(item_frames)) # batch_size * item_num * clip_num * emb_size
			frame_feats_value =torch.relu(self.frame_linear(item_frames))
		else:
			batch_size, item_num = item_ids.shape
			clip_num=40
		item_embed = self.item_embedding(item_ids)  # batch_size * item_num * emb_size
		item_embed_expanded = item_embed.unsqueeze(2).expand(-1, -1, clip_num, -1)  # batch_size * item_num * clip_num * emb_size
		
		## 2. frame position
		frame_positions = torch.arange(clip_num).unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1).float().to(item_ids.device)  # batch_size * item_num * clip_num
		frame_positions = frame_positions.unsqueeze(-1)
		frame_position_embed = self.frame_position_embedding(frame_positions)  # batch_size * item_num * clip_num * emb_size
		
		## 3. user embedding 
		user_ids = feed_dict['user_id']
		user_embed = self.user_embedding(user_ids)

		### 4.linear
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

	def _clip_integret_Rec(self, clip_predictions, feed_dict):
		batch_size, item_num, clip_num = clip_predictions.shape
		if self.adjust_interest_weight:
			interest_weight = self.trainable_interest_weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1) # batch_size*item_num*clip_num
		else:
			if 'c_interest_weight' in feed_dict:
				interest_weight = feed_dict['c_interest_weight'] # batch_size * item_num * clip_num(padding)
			else:
				interest_weight = torch.ones((batch_size, item_num, clip_num), device=clip_predictions.device)

		# mask
		item_duration = feed_dict['i_duration'] # batch_size * item_num; 代表每个视频有多少clip
		if self.duration_mask:
			mask = torch.arange(clip_num).unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1).to(item_duration.device) < item_duration.unsqueeze(-1)  # batch_size * item_num * clip_num
			# mask = mask.float()  # 转换为浮点型
		else:
			mask = torch.ones((batch_size, item_num, clip_num), device=item_duration.device).bool()
        
		if self.norm_interest_type=='softmax':
			'''
			interest_weight_normalized = F.softmax(interest_weight, dim=-1) * mask.float()
			'''
			# softmax_weights = F.softmax(interest_weight[mask == 1], dim=-1) 
			# interest_weight_normalized = torch.zeros_like(interest_weight)
			# interest_weight_normalized[mask] = softmax_weights
			interest_weight_masked = interest_weight.masked_fill(~mask, -float('inf'))
			interest_weight_normalized = F.softmax(interest_weight_masked, dim=-1)
		elif self.norm_interest_type=='sigmoid':
			interest_weight_normalized = F.sigmoid(interest_weight) * mask.float()
		else:
			interest_weight_normalized = interest_weight * mask.float()
		# import pdb; pdb.set_trace()
		weighted_predictions = clip_predictions * interest_weight_normalized
		predictions = weighted_predictions.sum(dim=-1)
		return predictions


	def forward(self, feed_dict):
		if 'i_item_frames' in feed_dict:
			user_embed, user_value, frame_feats_embed, frame_feats_value, frame_id_embed, frame_id_value = self._get_embeddings_ClipWDRec(feed_dict)
			frame_concat_embed = torch.cat([frame_feats_embed, frame_id_embed], dim=-1)  
			frame_concat_value = torch.cat([frame_feats_value,frame_id_value], dim=-1) 
        
		else:
			user_embed, user_value, frame_id_embed, frame_id_value = self._get_embeddings_ClipWDRec(feed_dict)
			frame_concat_embed = frame_id_embed
			frame_concat_value = frame_id_value
        
		batch_size, item_num, clip_num, _ = frame_concat_embed.shape
		user_embed = user_embed.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
		fm_vectors = torch.cat([user_embed, frame_concat_embed], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)
		deep_prediction = self.deep_layers(fm_vectors).squeeze(dim=-1)
		
		user_value = user_value.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
		linear_value = torch.cat([user_value, frame_concat_value], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)
		wide_prediction = self.overall_bias + linear_value.sum(dim=-1)
		clip_predictions = deep_prediction + wide_prediction

		predictions = self._clip_integret_Rec(clip_predictions, feed_dict)
		# TODO :先不管Contrastive和infoNCE Loss
		return {'prediction':predictions}
		
		
class ClipWDRecCTR(ContextCTRModel, ClipWDRecBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','loss_n','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
	@staticmethod
	def parse_model_args(parser):
		parser = ClipWDRecBase.parse_model_args_ClipWD(parser)
		return ContextCTRModel.parse_model_args(parser)
    
	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = ClipWDRecBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class ClipWDRecRanking(ContextModel,ClipWDRecBase):
	reader, runner = 'ContextReader','BaseRunner'
	extra_log_args = ['emb_size','layers','loss_n','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']

	@staticmethod
	def parse_model_args(parser):
		parser = ClipWDRecBase.parse_model_args_ClipWD(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args, corpus)

	def forward(self, feed_dict):
		return ClipWDRecBase.forward(self, feed_dict)
