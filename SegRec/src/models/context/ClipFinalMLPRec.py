"""
References:
	'FinalMLP: an enhanced two-stream MLP model for CTR prediction', Mao et al., AAAI2023.
Implementation reference: FuxiCTR
	https://github.com/reczoo/FuxiCTR/tree/v2.0.1/model_zoo/FinalMLP
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.ClipDCNv2Rec import ClipDCNv2RecBase
from utils.layers import MLP_Block
import torch.nn.functional as F

class ClipFinalMLPRecBase(ClipDCNv2RecBase):
	@staticmethod
	def parse_model_args_ClipFinalmlp(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--mlp1_hidden_units', type=str,default='[64,64,64]',
                      		help='Hidden units list of MLP1.')
		parser.add_argument('--mlp1_hidden_activations',type=str,default='ReLU',
                      		help='Hidden activation function of MLP1.')
		parser.add_argument('--mlp1_dropout',type=float,default=0,
                      		help='Dropout rate of MLP1.')
		parser.add_argument('--mlp1_batch_norm',type=int,default=0,
                      		help='Whether to use batch normalization in MLP1.')
		parser.add_argument('--mlp2_hidden_units', type=str,default='[64,64,64]',
                      		help='Hidden units list of MLP2.')
		parser.add_argument('--mlp2_hidden_activations',type=str,default='ReLU',
                      		help='Hidden activation function of MLP2.')
		parser.add_argument('--mlp2_dropout',type=float,default=0,
                      		help='Dropout rate of MLP2.')
		parser.add_argument('--mlp2_batch_norm',type=int,default=0,
                      		help='Whether to use batch normalization in MLP2.')
		parser.add_argument('--use_fs',type=int,default=1, help='Whether to use feature selection module.')
		parser.add_argument('--fs_hidden_units',type=str,default='[64]', help='Hidden units list of feature selection module.')
		parser.add_argument('--fs1_context',type=str,default='', help='Names of context features used for MLP1, split by commas.')
		parser.add_argument('--fs2_context',type=str,default='', help='Names of context features used for MLP2, split by commas.')
		parser.add_argument('--num_heads',type=int,default=1, help='Number of heads in the fusion module.')
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
		self.use_fs = args.use_fs
		self.fs1_context = [f for f in args.fs1_context.split(",") if len(f)]
		self.fs2_context = [f for f in args.fs2_context.split(",") if len(f)]
		
		self.contrastive = args.contrastive
		self.adjust_interest_weight = args.adjust_interest_weight
		self.duration_mask = args.duration_mask
		self.norm_interest_type = args.norm_interest_type

		self._define_params_ClipFinalmlp(args)
		self.apply(self.init_weights)

	def _define_params_ClipFinalmlp(self,args):
		# embeddings
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
		# MLP
		self.mlp1 = MLP_Block(input_dim=pre_size,output_dim=None,hidden_units=eval(args.mlp1_hidden_units),
						hidden_activations=args.mlp1_hidden_activations,dropout_rates=args.mlp1_dropout,
						batch_norm=args.mlp1_batch_norm)
		self.mlp2 = MLP_Block(input_dim=pre_size,output_dim=None,hidden_units=eval(args.mlp2_hidden_units),
						hidden_activations=args.mlp2_hidden_activations,dropout_rates=args.mlp2_dropout,
						batch_norm=args.mlp2_batch_norm)
		# fs
		if self.use_fs:
			self.fs_module = FeatureSelection({},pre_size,
									 self.vec_size, eval(args.fs_hidden_units),
									 self.fs1_context,self.fs2_context,self.feature_max)
		self.fusion_module = InteractionAggregation(eval(args.mlp1_hidden_units)[-1],
									eval(args.mlp2_hidden_units)[-1],output_dim=1,num_heads=args.num_heads)

	def _clip_integret_Rec_FinalMLP(self, clip_predictions, feed_dict):
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
			user_embed, frame_feats_embed, frame_id_embed = self._get_embeddings_ClipDCNv2(feed_dict)
			frame_concat_embed = torch.cat([frame_feats_embed, frame_id_embed], dim=-1)  
			
		else:
			user_embed, frame_id_embed = self._get_embeddings_ClipDCNv2(feed_dict)
			frame_concat_embed = frame_id_embed
		
		batch_size, item_num, clip_num, _ = frame_concat_embed.shape
		user_embed = user_embed.unsqueeze(1).unsqueeze(1).repeat(1, item_num, clip_num,1) # batch_size * item_num * clip_num * embedding_dim
		flat_emb = torch.cat([user_embed, frame_concat_embed], dim=-1) # batch_size * item_num * clip_num * (3*embedding_dim)

		if self.use_fs:
			feat1, feat2 = self.fs_module(feed_dict, flat_emb)
		else:
			feat1, feat2 = flat_emb, flat_emb
		emb_dim1, emb_dim2 = feat1.shape[-1], feat2.shape[-1]
		mlp1_output = self.mlp1(feat1.view(-1,emb_dim1)).view(batch_size, item_num, clip_num, -1)
		mlp2_output = self.mlp2(feat2.view(-1,emb_dim2)).view(batch_size, item_num, clip_num, -1)

		clip_predictions = self.fusion_module(mlp1_output, mlp2_output)
		
		predictions = self._clip_integret_Rec_FinalMLP(clip_predictions, feed_dict)

		return {'prediction':predictions}

class ClipFinalMLPRecCTR(ContextCTRModel, ClipFinalMLPRecBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n','use_fs','mlp1_hidden_units','mlp2_hidden_units','num_heads','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
    
	@staticmethod
	def parse_model_args(parser):
		parser = ClipFinalMLPRecBase.parse_model_args_ClipFinalmlp(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = ClipFinalMLPRecBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class ClipFinalMLPRecRanking(ContextModel, ClipFinalMLPRecBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','loss_n','use_fs','mlp1_hidden_units','mlp2_hidden_units','num_heads','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
    
	@staticmethod
	def parse_model_args(parser):
		parser = ClipFinalMLPRecBase.parse_model_args_ClipFinalmlp(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return ClipFinalMLPRecBase.forward(self, feed_dict)


class FeatureSelection(nn.Module):
	def __init__(self, feature_map, feature_dim, vec_size, fs_hidden_units=[], 
				 fs1_context=[], fs2_context=[],feature_maxn=dict()):
		super(FeatureSelection, self).__init__()
		self.fs1_context = fs1_context
		if len(fs1_context) == 0:
			self.fs1_ctx_bias = nn.Parameter(torch.zeros(1, vec_size))
		else:
			'''
			Reference:
				https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/embeddings/feature_embedding.py
			'''
			self.fs1_ctx_emb = []
			for ctx in fs1_context:
				if ctx.endswith("_c") and ctx in feature_maxn:
					self.fs1_ctx_emb.append(nn.Embedding(feature_maxn[ctx],vec_size))
				elif ctx.endswith("_f"):
					self.fs1_ctx_emb.append(nn.Linear(1,vec_size))
				else:
					raise ValueError("Undifined context %s"%(ctx))
			self.fs1_ctx_emb = nn.ModuleList(self.fs1_ctx_emb)
		self.fs2_context = fs2_context
		if len(fs2_context) == 0:
			self.fs2_ctx_bias = nn.Parameter(torch.zeros(1, vec_size))
		else:
			self.fs2_ctx_emb = []
			for ctx in fs2_context:
				if ctx.endswith("_c") and ctx in feature_maxn:
					self.fs2_ctx_emb.append(nn.Embedding(feature_maxn[ctx],vec_size))
				elif ctx.endswith("_f"):
					self.fs2_ctx_emb.append(nn.Linear(1,vec_size))
				else:
					raise ValueError("Undifined context %s"%(ctx))
			self.fs2_ctx_emb = nn.ModuleList(self.fs2_ctx_emb)
		self.fs1_gate = MLP_Block(input_dim=vec_size * max(1, len(fs1_context)),
								  output_dim=feature_dim,
								  hidden_units=fs_hidden_units,
								  hidden_activations="ReLU",
								  output_activation="Sigmoid",
								  batch_norm=False)
		self.fs2_gate = MLP_Block(input_dim=vec_size * max(1, len(fs2_context)),
								  output_dim=feature_dim,
								  hidden_units=fs_hidden_units,
								  hidden_activations="ReLU",
								  output_activation="Sigmoid",
								  batch_norm=False)
		
	def forward(self, feed_dict, flat_emb):
		
		if len(self.fs1_context) == 0:
			fs1_input = self.fs1_ctx_bias.unsqueeze(1).repeat(flat_emb.size(0),flat_emb.size(1),flat_emb.size(2), 1)
		else:
			fs1_input = []
			for i,ctx in enumerate(self.fs1_context):
				if ctx.endswith('_c'):
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx])
				else:
					ctx_emb = self.fs1_ctx_emb[i](feed_dict[ctx].float().unsqueeze(-1))
				if len(ctx_emb.shape)==2:
					fs1_input.append(ctx_emb.unsqueeze(1).repeat(1,flat_emb.size(1),flat_emb.size(2),1))
				else:
					fs1_input.append(ctx_emb)
			fs1_input = torch.cat(fs1_input,dim=-1)
		gt1 = self.fs1_gate(fs1_input) * 2
		feature1 = flat_emb * gt1
		if len(self.fs2_context) == 0:
			fs2_input = self.fs2_ctx_bias.unsqueeze(1).repeat(flat_emb.size(0),flat_emb.size(1),flat_emb.size(2), 1)
		else:
			fs2_input = []
			for i,ctx in enumerate(self.fs2_context):
				if ctx.endswith('_c'):
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx])
				else:
					ctx_emb = self.fs2_ctx_emb[i](feed_dict[ctx].float().unsqueeze(-1))
				if len(ctx_emb.shape)==2:
					fs2_input.append(ctx_emb.unsqueeze(1).repeat(1,flat_emb.size(1),flat_emb.size(2),1))
				else:
					fs2_input.append(ctx_emb)
			fs2_input = torch.cat(fs2_input,dim=-1)
		gt2 = self.fs2_gate(fs2_input) * 2
		feature2 = flat_emb * gt2
		return feature1, feature2

class InteractionAggregation(nn.Module):
	def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
		super(InteractionAggregation, self).__init__()
		assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
			"Input dim must be divisible by num_heads!"
		self.num_heads = num_heads
		self.output_dim = output_dim
		self.head_x_dim = x_dim // num_heads
		self.head_y_dim = y_dim // num_heads
		self.w_x = nn.Linear(x_dim, output_dim)
		self.w_y = nn.Linear(y_dim, output_dim)
		self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
											  output_dim))
		nn.init.xavier_normal_(self.w_xy)

	def forward(self, x, y):
		batch_size, item_num, clip_num, _ = x.shape
		output = self.w_x(x) + self.w_y(y)
		head_x = x.view(batch_size, item_num, clip_num, self.num_heads, self.head_x_dim).flatten(start_dim=0,end_dim=2)
		head_y = y.view(batch_size, item_num, clip_num, self.num_heads, self.head_y_dim).flatten(start_dim=0,end_dim=2)
		xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
									   self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
							   .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
						  head_y.unsqueeze(-1)).squeeze(-1)
		xy_reshape = xy.sum(dim=1).view(batch_size,item_num,clip_num, -1)
		output += xy_reshape
		return output.squeeze(-1)