"""AutoInt
Reference:
	Weiping Song et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks"
	in CIKM 2018.
Implementation reference: AutoInt and FuxiCTR
	https://github.com/shichence/AutoInt/blob/master/model.py
	https://github.com/reczoo/FuxiCTR/blob/main/model_zoo/AutoInt/src/AutoInt.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.ClipWDRec import ClipWDRecBase
from utils.layers import MultiHeadAttention, MLP_Block
import torch.nn.functional as F

class ClipAutoIntRecBase(ClipWDRecBase):
	@staticmethod
	def parse_model_args_ClipAutoInt(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--attention_size', type=int, default=32,
							help='Size of attention hidden space.')
		parser.add_argument('--num_heads', type=int, default=1,
							help='Number of attention heads.')
		parser.add_argument('--num_layers', type=int, default=1,
							help='Number of self-attention layers.') # for attention layer
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.") # for deep layers
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
		self.vec_size = args.emb_size
		self.layers = eval(args.layers)
  
		self.num_heads = args.num_heads
		self.num_layers = args.num_layers
		self.attention_size= args.attention_size

		self.contrastive = args.contrastive
		self.adjust_interest_weight = args.adjust_interest_weight
		self.duration_mask = args.duration_mask
		self.norm_interest_type = args.norm_interest_type

		self._define_params_ClipAutoInt()
		self.apply(self.init_weights)

	def _define_params_ClipAutoInt(self):
		self._define_params_ClipWD()
		# Attention
		att_input = self.vec_size
		autoint_attentions = []
		residual_embeddings = []
		for _ in range(self.num_layers):
			autoint_attentions.append(
				MultiHeadAttention(d_model=att_input, n_heads=self.num_heads, kq_same=False, bias=False,
									attention_d=self.attention_size))
			residual_embeddings.append(nn.Linear(att_input, self.attention_size))
			att_input = self.attention_size
		self.autoint_attentions = nn.ModuleList(autoint_attentions)
		self.residual_embeddings = nn.ModuleList(residual_embeddings)
		# Deep
		pre_size = len(self.feature_max) * self.attention_size
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   dropout_rates=self.dropout, output_dim=1)

	def forward(self, feed_dict):
		# autoint_all_embeddings, linear_value = self._get_embeddings_FM(feed_dict)
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
        # print('user_embed',user_embed.shape)
		fm_vectors = torch.cat([user_embed, frame_concat_embed], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)
        
		user_value = user_value.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
		linear_value = torch.cat([user_value, frame_concat_value], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)
        
		autoint_all_embeddings = fm_vectors.reshape(batch_size, item_num, clip_num, -1, self.vec_size)
		linear_value = self.overall_bias + linear_value.sum(dim=-1)

		# Attention + Residual
		for autoint_attention, residual_embedding in zip(self.autoint_attentions, self.residual_embeddings):
			attention = autoint_attention(autoint_all_embeddings, autoint_all_embeddings, autoint_all_embeddings)
			residual = residual_embedding(autoint_all_embeddings)
			autoint_all_embeddings = (attention + residual).relu()
		# Deep
		deep_vectors = autoint_all_embeddings.flatten(start_dim=-2)
		deep_vectors = self.deep_layers(deep_vectors)
		clip_predictions = linear_value + deep_vectors.squeeze(-1)

		predictions = self._clip_integret_Rec(clip_predictions, feed_dict)

		return {'prediction':predictions}

class ClipAutoIntRecCTR(ContextCTRModel, ClipAutoIntRecBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','layers','num_layers','num_heads','contrastive', 'loss_n','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipAutoIntRecBase.parse_model_args_ClipAutoInt(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)
	
	def forward(self, feed_dict):
		out_dict = ClipAutoIntRecBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict	

class ClipAutoIntRecRanking(ContextModel,ClipAutoIntRecBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','layers','num_layers','num_heads','loss_n','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipAutoIntRecBase.parse_model_args_ClipAutoInt(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)
	
	def forward(self, feed_dict):
		return ClipAutoIntRecBase.forward(self, feed_dict)
