
""" FM
Reference:
	'Factorization Machines', Steffen Rendle, 2010 IEEE International conference on data mining.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel

class FMBase(object):
	@staticmethod
	def parse_model_args_FM(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _define_init_params(self, args,corpus):
		self.vec_size = args.emb_size
		self._define_params_FM()
		self.apply(self.init_weights)
	
	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self._define_params_FM()
		self.apply(self.init_weights)
	
	def _define_params_FM(self):	
		self.context_embedding = nn.ModuleDict()
		self.linear_embedding = nn.ModuleDict()
		for f in self.context_features:
			self.context_embedding[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)
			self.linear_embedding[f] = nn.Embedding(self.feature_max[f],1) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,1,bias=False)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

		if len(self.clip_feature_path):
			self.item_frames_context_embedding = nn.Linear(1024, self.vec_size, bias=False) # reader 占有存储空间的策略
			self.item_frames_linear_embedding = nn.Linear(1024, 1, bias=False)

	def _get_embeddings_FM(self, feed_dict):
		item_ids = feed_dict['item_id']
		_, item_num = item_ids.shape

		fm_vectors = [self.context_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.context_embedding[f](feed_dict[f].float().unsqueeze(-1)) for f in self.context_features]
		fm_vectors = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in fm_vectors], dim=-2) # batch size * item num * feature num * feature dim: 84,100,2,64
		linear_value = [self.linear_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
							else self.linear_embedding[f](feed_dict[f].float().unsqueeze(-1)) for f in self.context_features]
		linear_value = torch.cat([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1)
	  				for v in linear_value],dim=-1) # batch size * item num * feature num
		if len(self.clip_feature_path):
			item_frames_context_emb = self.item_frames_context_embedding(feed_dict['i_item_frames']).float() # batch * *item num * clip num * emb size
			item_frames_linear_emb = self.item_frames_linear_embedding(feed_dict['i_item_frames']).float().squeeze(-1) # batch *item num * clip num * emb size
			fm_vectors = torch.cat((fm_vectors,item_frames_context_emb),dim=-2)
			linear_value = torch.cat((linear_value,item_frames_linear_emb),dim=-1)
		linear_value = self.overall_bias + linear_value.sum(dim=-1)
		return fm_vectors, linear_value

	def forward(self, feed_dict):
		fm_vectors, linear_value = self._get_embeddings_FM(feed_dict)
		import pdb; pdb.set_trace()
		fm_vectors = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2))
		predictions = linear_value + fm_vectors.sum(dim=-1)
		return {'prediction':predictions}

class FMCTR(ContextCTRModel, FMBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = FMBase.parse_model_args_FM(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		out_dict = FMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class FMRanking(ContextModel,FMBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = FMBase.parse_model_args_FM(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)

	def forward(self, feed_dict):
		return FMBase.forward(self, feed_dict)