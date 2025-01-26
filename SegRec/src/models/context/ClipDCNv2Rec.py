""" DCN v2
Reference:
	'DCN v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems.', Wang et al, WWW2021.
Implementation reference: RecBole
	https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcnv2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.DCN import DCNBase
from models.context.ClipWDRec import ClipWDRecBase
from utils.layers import MLP_Block
import torch.nn.functional as F

class ClipDCNv2RecBase(ClipWDRecBase):
	@staticmethod
	def parse_model_args_ClipDCNv2(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each deep layer.")
		parser.add_argument('--cross_layer_num',type=int,default=6,
							help="Number of cross layers.")
		parser.add_argument('--reg_weight',type=float, default=2.0,
                      		help="Regularization weight for cross-layer weights. In DCNv2, it is only used for mixed version")
		parser.add_argument('--mixed',type=int, default=1,
                            help="Wether user mixed cross network or not.")
		parser.add_argument('--structure',type=str, default='parallel',
                            help="cross network and DNN is 'parallel' or 'stacked'")
		parser.add_argument('--low_rank',type=int, default=64,
                            help="Size for the low-rank architecture when mixed==1.")
		parser.add_argument('--expert_num', type=int, default=2,
                            help="Number of experts to calculate in each cross layer when mixed==1.")
		parser.add_argument('--contrastive', type=str, default='',
							help='Choose the contrastive loss if used')
		parser.add_argument('--adjust_interest_weight', type=int, default=0,
							help="Whether to adjust interest weight or not.")
		parser.add_argument('--duration_mask', type=int, default=0,
                            help="Whether to get duration mask or not.")
		parser.add_argument('--norm_interest_type', type=str, default='none', choices=['softmax', 'sigmoid','none'], 
                            help="Which normalized type to use on interest logits.") # original:softmax_interest,but now Do on logit
		return parser

	def _define_init(self, args, corpus):
		# self._define_init_params(args,corpus) # DCN
		self.vec_size = args.emb_size
		self.reg_weight = args.reg_weight
		self.layers = eval(args.layers)
		self.cross_layer_num = args.cross_layer_num
		
		self.mixed = args.mixed
		self.structure = args.structure
		self.expert_num = args.expert_num
		self.low_rank = args.low_rank

		self.contrastive = args.contrastive
		self.adjust_interest_weight = args.adjust_interest_weight
		self.duration_mask = args.duration_mask
		self.norm_interest_type = args.norm_interest_type

		self._define_params_DCNv2()
		self.apply(self.init_weights)

	def _define_params_DCNv2(self):
        # embedding
		self.user_embedding = nn.Embedding(self.feature_max['user_id'], self.vec_size)
		self.item_embedding = nn.Embedding(self.feature_max['item_id'], self.vec_size)
		self.frame_position_embedding = nn.Linear(1, self.vec_size)
		if len(self.clip_feature_path):
			self.frame_feature_dim=1024
			self.frame_embedding = nn.Linear(self.frame_feature_dim, self.vec_size)
			self.frame_id_projector = nn.Linear(self.vec_size*2, self.vec_size)
        
		if self.adjust_interest_weight:
			self.trainable_interest_weight = torch.nn.Parameter(torch.ones(40), requires_grad=True) # 这里是对所有交互固定的clip概率，而不是f(user item time)
		
		pre_size = len(self.feature_max) * self.vec_size
		# cross layers
		if self.mixed:
			self.cross_layer_u = nn.ParameterList(nn.Parameter(torch.randn(self.expert_num, pre_size, self.low_rank))
                            for l in range(self.cross_layer_num)) # U: (pre_size, low_rank)
			self.cross_layer_v = nn.ParameterList(nn.Parameter(torch.randn(self.expert_num, pre_size, self.low_rank))
                            for l in range(self.cross_layer_num)) # V: (pre_size, low_rank)
			self.cross_layer_c = nn.ParameterList(nn.Parameter(torch.randn(self.expert_num, self.low_rank, self.low_rank))
                            for l in range(self.cross_layer_num)) # V: (pre_size, low_rank)
			self.gating = nn.ModuleList(nn.Linear(pre_size, 1) for l in range(self.expert_num))
		else:
			self.cross_layer_w2 = nn.ParameterList(nn.Parameter(torch.randn(pre_size, pre_size))
                			for l in range(self.cross_layer_num)) # W: (pre_size, pre_size)
		self.bias = nn.ParameterList(nn.Parameter(torch.zeros(pre_size, 1))for l in range(self.cross_layer_num))
		self.tanh = nn.Tanh()

		# deep layers
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   batch_norm=True,norm_before_activation=True,
          						dropout_rates=self.dropout, output_dim=None)
		# output layer
		if self.structure == "parallel":
			self.predict_layer = nn.Linear(len(self.feature_max) * self.vec_size + self.layers[-1], 1)
		if self.structure == "stacked":
			self.predict_layer = nn.Linear(self.layers[-1], 1)
		
	def cross_net_2(self, x_0):
		# x_0: batch size * item num * embedding size
		'''
		math:: x_{l+1} = x_0 * {W_l · x_l + b_l} + x_l
        '''
		batch_size, item_num, clip_num, output_emb = x_0.shape
		x_0 = x_0.view(-1,  output_emb)
		x_0 = x_0.unsqueeze(dim=2)
		x_l = x_0 # x0: (batch_size * num_item, context_num * emb_size, 1)
		for layer in range(self.cross_layer_num): # self.cross_layer_w2[layer]: (context_num * emb_size, context_num * emb_size)
			xl_w = torch.matmul(self.cross_layer_w2[layer], x_l) # wl_w: (batch_size * num_item, context_num * emb_size, 1)
			xl_w = xl_w + self.bias[layer] # self.bias[layer]: (context_num * emb_size, 1)
			xl_dot = torch.mul(x_0, xl_w) # wl_dot: (batch_size * num_item, context_num * emb_size, 1)
			x_l = xl_dot + x_l
		x_l = x_l.view(batch_size, item_num, clip_num, -1)
		return x_l

	def cross_net_mix(self, x_0):
		"""Reference: RecBole - https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/context_aware_recommender/dcnv2.py
		add MoE and nonlinear transformation in low-rank space
        .. math::
            x_{l+1} = \sum_{i=1}^K G_i(x_l)E_i(x_l)+x_l
        .. math::
            E_i(x_l) = x_0 \odot (U_l^i \dot g(C_l^i \dot g(V_L^{iT} x_l)) + b_l)

        :math:`E_i` and :math:`G_i` represents the expert and gatings respectively,
        :math:`U_l`, :math:`C_l`, :math:`V_l` stand for low-rank decomposition of weight matrix,
        :math:`g` is the nonlinear activation function.

        Args:
            x_0(torch.Tensor): Embedding vectors of all features, input of cross network.

        Returns:
            torch.Tensor:output of mixed cross network, [batch_size, num_feature_field * embedding_size]
        """
		# x_0: batch size * item num * embedding size
		batch_size, item_num, clip_num, output_emb = x_0.shape
		x_0 = x_0.view(-1,  output_emb) # x0: (batch_size * num_item, context_num * emb_size)
		x_0 = x_0.unsqueeze(dim=2)
		x_l = x_0 # x0: (batch_size * num_item, context_num * emb_size, 1)
		for layer in range(self.cross_layer_num):
			expert_output_list = []
			gating_output_list = []
			for expert in range(self.expert_num):
                # compute gating output
				gating_output_list.append(self.gating[expert](x_l.squeeze(dim=2)))  # (batch_size, 1)
                # project to low-rank subspace
				xl_v = torch.matmul(self.cross_layer_v[layer][expert].T, x_l)  # (batch_size, low_rank, 1)
                # nonlinear activation in subspace
				xl_c = nn.Tanh()(xl_v)
				xl_c = torch.matmul(self.cross_layer_c[layer][expert], xl_c)  # (batch_size, low_rank, 1)
				xl_c = nn.Tanh()(xl_c)
                # project back feature space
				xl_u = torch.matmul(self.cross_layer_u[layer][expert], xl_c)  # (batch_size, in_feature_num, 1)
                # dot with x_0
				xl_dot = xl_u + self.bias[layer]
				xl_dot = torch.mul(x_0, xl_dot)
				expert_output_list.append(xl_dot.squeeze(dim=2))  # (batch_size, in_feature_num)

			expert_output = torch.stack(expert_output_list, dim=2)  # (batch_size, in_feature_num, expert_num)
			gating_output = torch.stack(gating_output_list, dim=1)  # (batch_size, expert_num, 1)
			moe_output = torch.matmul(expert_output, nn.Softmax(dim=1)(gating_output))  # (batch_size, in_feature_num, 1)
			x_l = x_l + moe_output

		x_l = x_l.view(batch_size, item_num, clip_num, -1)
		return x_l

	def _get_embeddings_ClipDCNv2(self, feed_dict):
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

		if 'i_item_frames' in feed_dict:
			frame_id_embed = self.frame_id_projector(torch.cat([item_embed_expanded, frame_position_embed], dim=-1)) # batch_size * item_num * clip_num * embedding_dim
			return user_embed, frame_feats_embed, frame_id_embed
		else:
			frame_id_embed = torch.cat([item_embed_expanded, frame_position_embed], dim=-1) # batch_size * item_num * clip_num * (2*embedding_dim)
			return user_embed, frame_id_embed

	def _clip_integret_Rec_DCNv2(self, clip_predictions, feed_dict):
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
		if 'i_item_frames' in feed_dict:
			user_embed, frame_feats_embed, frame_id_embed = self._get_embeddings_ClipDCNv2(feed_dict)
			frame_concat_embed = torch.cat([frame_feats_embed, frame_id_embed], dim=-1)  
			
		else:
			user_embed, frame_id_embed = self._get_embeddings_ClipDCNv2(feed_dict)
			frame_concat_embed = frame_id_embed

		batch_size, item_num, clip_num, _ = frame_concat_embed.shape
		user_embed = user_embed.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
		context_emb = torch.cat([user_embed, frame_concat_embed], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)

		if self.mixed:
			cross_output = self.cross_net_mix(context_emb) 
		else:
			cross_output = self.cross_net_2(context_emb)
		output_emb = cross_output.shape[-1] # corss_output: batch_size * item_num * clip_num * (3*embedding_dim)

		if self.structure == 'parallel':
			deep_output = context_emb.view(-1,output_emb) # (batch_size * num_item, context_num * emb_size)
			deep_output = self.deep_layers(deep_output).view(batch_size, item_num, clip_num, self.layers[-1]) # (batch_size, num_item, context_num * emb_size)
			output = self.predict_layer(torch.cat([cross_output, deep_output],dim=-1)) 
		elif self.structure == 'stacked':
			deep_output = cross_output.view(-1,output_emb)
			deep_output = self.deep_layers(deep_output).view(batch_size, item_num, clip_num, self.layers[-1]) # (batch_size, num_item, context_num * emb_size)
			output = self.predict_layer(deep_output)

		clip_predictions = output.squeeze(dim=-1)
		predictions = self._clip_integret_Rec_DCNv2(clip_predictions, feed_dict)

		return {'prediction':predictions}
		
class ClipDCNv2RecCTR(ContextCTRModel, ClipDCNv2RecBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','loss_n','cross_layer_num','structure','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipDCNv2RecBase.parse_model_args_ClipDCNv2(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = ClipDCNv2RecBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

	def loss(self, out_dict: dict):
		loss = ContextCTRModel.loss(self, out_dict)
		if not self.mixed:
			l2_loss = self.reg_weight * ClipDCNv2RecBase.l2_reg(self, self.cross_layer_w2)
			return loss + l2_loss
		else:
			return loss

class ClipDCNv2RecRanking(ContextModel, ClipDCNv2RecBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','layers','loss_n','cross_layer_num','structure','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipDCNv2RecBase.parse_model_args_ClipDCNv2(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return ClipDCNv2RecBase.forward(self, feed_dict)
	
	def loss(self, out_dict: dict):
		loss = ContextModel.loss(self, out_dict)
		if not self.mixed:
			l2_loss = self.reg_weight * ClipDCNv2RecBase.l2_reg(self, self.cross_layer_w2)
			return loss + l2_loss
		else:
			return loss