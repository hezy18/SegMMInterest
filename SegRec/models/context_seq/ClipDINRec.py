
import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from utils.layers import MLP_Block
import torch.nn.functional as F

class ClipDINRecBase(object):
	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer in the attention module.")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer in the MLP module.")
		parser.add_argument('--adjust_interest_weight', type=int, default=0,
							help="Whether to adjust interest weight or not.")
		parser.add_argument('--train_module', type=int, default=0, # 在这里尝试，但是没在其他backbone里用
							help="Whether to train module for interest_weight or not.")
		parser.add_argument('--duration_mask', type=int, default=0,
							help="Whether to get duration mask or not.")
		parser.add_argument('--contrastive', type=str, default='',
                            help='Choose the contrastive loss if used')
		parser.add_argument('--norm_interest_type', type=str, default='none', choices=['softmax', 'sigmoid','none'], 
							help="Which normalized type to use on interest logits.") # original:softmax_interest,but now Do on logit
		return parser

	def _define_init(self, args,corpus):
		self.user_context = ['user_id']+corpus.user_feature_names
		self.item_context = ['item_id']+corpus.item_feature_names
		self.situation_context = corpus.situation_feature_names
		self.item_feature_num = len(self.item_context)
		self.user_feature_num = len(self.user_context)
		self.situation_feature_num = len(corpus.situation_feature_names) if self.add_historical_situations else 0
  
		self.vec_size = args.emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)

		self.contrastive = args.contrastive
		self.adjust_interest_weight = args.adjust_interest_weight
		if self.adjust_interest_weight:
			self.trainable_interest_weight = torch.nn.Parameter(torch.ones(40), requires_grad=True) # 这里是对所有交互固定的clip概率，而不是f(user item time)
		self.duration_mask = args.duration_mask
		self.norm_interest_type = args.norm_interest_type
		self.train_module = args.train_module

		self._define_params_ClipDIN()
		self.apply(self.init_weights)

		

	def _define_params_ClipDIN(self):
		# for embedding:
		self.user_embedding = nn.Embedding(self.feature_max['user_id'], self.vec_size)
		self.item_embedding = nn.Embedding(self.feature_max['item_id'], self.vec_size)
		self.item_feature_embedding = nn.Linear(len(self.item_context)-1, self.vec_size)	

		self.frame_position_embedding = nn.Linear(1, self.vec_size)
		if len(self.clip_feature_path):
			self.frame_feature_dim=1024
			self.frame_embedding = nn.Linear(self.frame_feature_dim, self.vec_size)
		self.frame_id_projector = nn.Linear(self.vec_size*2, self.vec_size)
		
		# DIN
		self.item_feature_num=2
		self.situation_feature_num=0
		self.user_feature_num=1
		pre_size = 4 * (self.item_feature_num+self.situation_feature_num) * self.vec_size
		self.att_mlp_layers = MLP_Block(input_dim=pre_size,hidden_units=self.att_layers,output_dim=1,
                                  hidden_activations='Sigmoid',dropout_rates=self.dropout,batch_norm=False)

		pre_size = (2*(self.item_feature_num+self.situation_feature_num)
					+self.item_feature_num
              		#+len(self.situation_context) 
			  		+ self.user_feature_num) * self.vec_size
		self.dnn_mlp_layers = MLP_Block(input_dim=pre_size,hidden_units=self.dnn_layers,output_dim=1,
                                  hidden_activations='Dice',dropout_rates=self.dropout,batch_norm=True,
                                  norm_before_activation=True)

		if self.train_module:
			self.user_linear_interst = nn.Linear(self.vec_size, 1)
			self.item_linear_interest = nn.Linear(self.vec_size, 1)

	def attention(self, queries, keys, keys_length, mask_mat,softmax_stag=False, return_seq_weight=False):
		'''Reference:
			RecBole layers: SequenceAttLayer, https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/layers.py#L294
			queries: batch * (if*vecsize)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.repeat(1, hist_len)
		queries = queries.view(-1, hist_len, embedding_size)
		# MLP Layer
		input_tensor = torch.cat([queries, keys, queries - keys, queries * keys], dim=-1)
		# input_tensor 10240, 20, 512
		output = torch.transpose(self.att_mlp_layers(input_tensor), -1, -2)
		output = output.squeeze(1) # 10240,20
		# get mask
		mask = mask_mat.repeat(output.size(0), 1) # 10240,20
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
		output = output.unsqueeze(1)
		output = output / (embedding_size**0.5)
		# get the weight of each user's history list about the target item
		if softmax_stag:
			output = fn.softmax(output, dim=2)  # [B, 1, T]
		if not return_seq_weight:
			output = torch.matmul(output, keys)  # [B, 1, H]
		torch.cuda.empty_cache()
		return output.squeeze(dim=1)

	def _get_all_embedding_ClipDIN(self, feed_dict, merge_all=True):
		## 1. item embedding
		item_ids = feed_dict['item_id'] # batch_size * item_num
		if 'i_item_frames' in feed_dict:
			item_frames = feed_dict['i_item_frames'] # batch_size * item_num * clip_num(padding) * feature_dim(1024)
			batch_size, item_num, clip_num, _ = item_frames.shape
			frame_feats_embed = torch.relu(self.frame_embedding(item_frames)) # batch_size * item_num * clip_num * emb_size
		else:
			batch_size, item_num = item_ids.shape
			clip_num=40
		frame_positions = torch.arange(clip_num).unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1).float().to(item_ids.device)  # batch_size * item_num * clip_num
		item_embed = self.item_embedding(item_ids)  # batch_size * item_num * emb_size
		item_embed_expanded = item_embed.unsqueeze(2).expand(-1, -1, clip_num, -1)  # batch_size * item_num * clip_num * emb_size
		frame_positions = frame_positions.unsqueeze(-1)
		frame_position_embed = self.frame_position_embedding(frame_positions)  # batch_size * item_num * clip_num * emb_size
		if 'i_item_frames' in feed_dict:
			frame_id_embed = self.frame_id_projector(torch.cat([item_embed_expanded, frame_position_embed], dim=-1)) # batch_size * item_num * clip_num * emb_size
			frame_concat_embed = torch.cat([frame_feats_embed, frame_id_embed], dim=-1)  
		else:
			frame_id_embed = torch.cat([item_embed_expanded, frame_position_embed], dim=-1) # batch_size * item_num * clip_num * (2*emb_size)
			frame_concat_embed = frame_id_embed
		item_feats_emb = self.frame_id_projector(frame_concat_embed) # batch_size * item_num * clip_num * (2*emb_size)
		
		## 2. historical item embedding
		history_item_emb = self.item_embedding(feed_dict['history_item_id'] ) # batch * history_max * emb_size
		if 'i_duration' in feed_dict:
			item_feature_emb = self.item_feature_embedding(feed_dict['i_duration'].float().unsqueeze(-1))
			item_feats_emb = torch.cat([item_feats_emb, item_feature_emb.unsqueeze(2).expand(-1, -1, clip_num, -1)], dim=-1)
			history_feature_emb = self.item_feature_embedding(feed_dict['history_i_duration'].float().unsqueeze(-1))
			history_item_emb = torch.cat([history_item_emb, history_feature_emb], dim=-1)
		
		## 3. user embedding
		user_ids = feed_dict['user_id'] # batch_size
		user_embed = self.user_embedding(user_ids) # batch_size * emb_size

		user_feats_emb = user_embed

		# situation embedding
		# if len(self.situation_context):
			
		# situ_feats_emb = []
		# historical situation embedding
		# if self.add_historical_situations:
			
		history_emb = history_item_emb #.flatten(start_dim=-2)
		current_emb = item_feats_emb# .flatten(start_dim=-2)
		if merge_all:
			item_num = item_feats_emb.shape[1]
			# if len(situ_feats_emb):
			all_context = torch.cat([item_feats_emb, user_feats_emb.unsqueeze(1).unsqueeze(1).repeat(1,item_num,clip_num,1),
						],dim=-1) #.flatten(start_dim=-2)
					
			return history_emb, current_emb, all_context, item_feats_emb, user_feats_emb
		else:
			return history_emb, current_emb, item_feats_emb, user_feats_emb
		'''
		user_feats_emb # bsz,64
		history_emb # bsz,20
		item_feats_emb # bsz,1,40,128
		all_context # bsz,1,40,64+128
		current_emb # bsz,1,40,128
		'''

	def att_dnn(self, current_emb, history_emb, all_context, history_lengths):

		his_mask_mat = (torch.arange(history_emb.shape[1]).view(1,-1)).to(self.device) # 1, hislen

		# batch_size, item_num, feats_emb = current_emb.shape
		# _, max_len, his_emb = history_emb.shape

		batch_size, item_num, clip_num, feats_emb = current_emb.shape
		_, max_len, his_emb = history_emb.shape

		current_emb2d = current_emb.view(-1, feats_emb) # transfer 4d (batch_size * item_num * clip_num * feats_emb) to 2d ((batch_size*item_num*clip_num) * feats_emb) 
		# current_emb2d 10240,128
		history_emb_unsqueeze = history_emb.unsqueeze(1).unsqueeze(1).repeat(1,item_num,clip_num,1,1)
		history_emb2d = history_emb_unsqueeze.view(-1,max_len,his_emb)
		# history_emb2d 10240,20,64
		hislens2d = history_lengths.unsqueeze(1).unsqueeze(1).repeat(1,item_num, clip_num).view(-1)
		# hislens2d 10240
		user_his_emb2d = self.attention(current_emb2d, history_emb2d, hislens2d, his_mask_mat, softmax_stag=False)
		# user_his_emb2d 10240,64
		din_output = torch.cat([user_his_emb2d, user_his_emb2d*current_emb2d, all_context.view(batch_size*item_num*clip_num,-1) ],dim=-1)
		# din_output 10240,488
		din_output = self.dnn_mlp_layers(din_output)
		return din_output.squeeze(dim=-1).view(batch_size, item_num, clip_num)

	def _clip_integret_Rec(self, clip_predictions, feed_dict):
		batch_size, item_num, clip_num = clip_predictions.shape
		if self.adjust_interest_weight:
			interest_weight = self.trainable_interest_weight.unsqueeze(0).unsqueeze(0).repeat(batch_size, item_num, 1) # batch_size*item_num*clip_num
		# elif self.train_module:
		# 	user_value = self.user_linear_interst(user_feats_emb).unsqueeze(-1)
		# 	item_value = self.item_linear_interest(item_feats_emb).squeeze(-1)
		# 	interest_weight = item_value * user_value
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

		weighted_predictions = clip_predictions * interest_weight_normalized
		predictions = weighted_predictions.sum(dim=-1)

		return predictions
	
	def forward(self, feed_dict):
		hislens = feed_dict['lengths'] # batch_size
		
		history_emb, current_emb, all_context, item_feats_emb, user_feats_emb = self._get_all_embedding_ClipDIN(feed_dict)
		clip_predictions = self.att_dnn(current_emb,history_emb, all_context, hislens)
		# clip_predictions: batch_size * item_num * clip_num
		
		predictions = self._clip_integret_Rec(clip_predictions, feed_dict)

		return {'prediction':predictions}


class ClipDINRecCTR(ContextSeqCTRModel, ClipDINRecBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','att_layers','dnn_layers','duration_mask','norm_interest_type','adjust_interest_weight','clip_weight_path','train_module']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipDINRecBase.parse_model_args(parser)
		return ContextSeqCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = ClipDINRecBase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class ClipDINRecRanking(ContextSeqModel, ClipDINRecBase):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size','att_layers','dnn_layers','duration_mask','norm_interest_type','adjust_interest_weight','clip_weight_path','train_module']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipDINRecBase.parse_model_args(parser)
		return ContextSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return ClipDINRecBase.forward(self, feed_dict)
