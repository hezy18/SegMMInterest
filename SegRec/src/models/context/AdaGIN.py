"""AdaGIN / AutoGIM 
Reference:
	AdaGIN: Adaptive Graph Interaction Network for Click-Through Rate Prediction
	in TOIS 2024.
    (old name) AutoGIM: Autonomous Graph Interaction Machine
Implementation reference: AdaGIN and FuxiCTR
	https://github.com/salmon1802/AdaGIN/blob/main/AutoGIM/src/AutoGIM.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel
from models.context.FM import FMBase
from utils.layers import MultiHeadAttention, MLP_Block
import torch.nn.functional as F

class AdaGINBase(FMBase):
	@staticmethod
	def parse_model_args_AdaGIN(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--warm_dim', type=int, default=64,
							help='')
		parser.add_argument('--cold_dim', type=int, default=64,
							help='')
		parser.add_argument('--warm_tau', type=float, default=1.0,
							help='')
		parser.add_argument('--cold_tau', type=float, default=0.01,
							help='')
		parser.add_argument('--fi_hidden_units', type=str, default='[64,64]',
							help='Layer Size for fi.')
		parser.add_argument('--w_hidden_units', type=str, default='[64,64]',
							help='Layer Size for w.') # TODO：mark ！该参数不影响结果，调参数先不管
		parser.add_argument('--hidden_activation', type=str, default='ReLU', # leaky_relu
							help='Hidden activation.')
		parser.add_argument('--num_gnn_layers', type=int, default=3,
							help='Number of GNN layers.')
		parser.add_argument('--only_use_last_layer', type=int, default=1,
							help='Whether to only use last layer.')

		return parser 
	
	def _define_init(self, args, corpus):
		self.vec_size = args.emb_size
		self.fi_hidden_units = eval(args.fi_hidden_units)
		self.w_hidden_units = eval(args.w_hidden_units)
		self.hidden_activation = args.hidden_activation
  
		self.warm_dim = args.warm_dim
		self.cold_dim = args.cold_dim
		self.warm_tau= args.warm_tau
		self.cold_tau= args.cold_tau

		self.num_gnn_layers= args.num_gnn_layers
		self.only_use_last_layer = args.only_use_last_layer

		self._define_params_AdaGIN()
		self.apply(self.init_weights)

	def _define_params_AdaGIN(self):
		self._define_params_FM()
		# hidden units
		self.pre_ep_size = int(len(self.feature_max)  * (len(self.feature_max)+1) / 2) * self.vec_size
		self.mlp1 = MLP_Block(input_dim=self.pre_ep_size, output_dim=1, hidden_units = self.fi_hidden_units, 
							hidden_activations = self.hidden_activation, dropout_rates=self.dropout)
		self.W1 = MLP_Block(input_dim=self.pre_ep_size, output_dim=1, hidden_units = self.w_hidden_units, 
							hidden_activations = self.hidden_activation, 
							output_activation = 'LeakyReLU', dropout_rates=self.dropout)

		self.pre_ip_size = int(len(self.feature_max)  * (len(self.feature_max)+1) / 2) 
		self.mlp2 = MLP_Block(input_dim=self.pre_ip_size, output_dim=1, hidden_units = self.fi_hidden_units, 
							hidden_activations = self.hidden_activation, dropout_rates=self.dropout)
		self.W2 = MLP_Block(input_dim=self.pre_ip_size, output_dim=1, hidden_units=self.w_hidden_units, 
							hidden_activations = self.hidden_activation, 
							output_activation = 'LeakyReLU', dropout_rates=self.dropout)

		self.pre_flatten_size = int(len(self.feature_max)) * self.vec_size
		self.mlp3 = MLP_Block(input_dim=self.pre_flatten_size, output_dim=1, hidden_units=self.fi_hidden_units,
							hidden_activations = self.hidden_activation, dropout_rates=self.dropout)
		self.W3 = MLP_Block(input_dim=self.pre_flatten_size, output_dim=1, hidden_units =self.w_hidden_units, 
							hidden_activations = self.hidden_activation, 
							output_activation = 'LeakyReLU', dropout_rates=self.dropout)

		self.triu_node_index = nn.Parameter(torch.triu_indices(len(self.feature_max), len(self.feature_max), offset=0),
											requires_grad=False)
		# graph
		self.AutoGraph = AutoGraph_Layer(num_fields=len(self.feature_max),
											embedding_dim=self.vec_size,
											warm_dim=self.warm_dim,
											cold_dim=self.cold_dim,
											warm_tau=self.warm_tau,
											cold_tau=self.cold_tau,
											only_use_last_layer=self.only_use_last_layer,
											gnn_layers=self.num_gnn_layers)
		# final
		final_score_weight = torch.rand(self.num_gnn_layers)
		final_score_weight = final_score_weight / torch.sum(final_score_weight)
		self.final_score_weight = nn.Parameter(final_score_weight, requires_grad=True)


	def ep_ip_flatten_emb(self, h):
		emb1 = torch.index_select(h, -2, self.triu_node_index[0])
		emb2 = torch.index_select(h, -2, self.triu_node_index[1])
		embs_ep = emb1 * emb2
		embs_ip = torch.sum(embs_ep, dim=-1) # bsz*n_item, pre_ep_size/emb_dim
		embs_ep = embs_ep.view(-1, self.pre_ep_size) # bsz*n_item, pre_ep_size
		embs_flatten = h.flatten(start_dim=1) # bsz*n_item, n_feat*emb_dim
		return [embs_ep, embs_ip, embs_flatten]
    
	def all_product(self, embs_list):
		W1 = self.W1(embs_list[0])
		X1 = self.mlp1(embs_list[0])
		W1X1 = W1 * X1
		W2 = self.W2(embs_list[1])
		X2 = self.mlp2(embs_list[1])
		W2X2 = W2 * X2
		W3 = self.W3(embs_list[2])
		X3 = self.mlp3(embs_list[2])
		W3X3 = W3 * X3
		return [W1X1, W2X2, W3X3]

	def forward(self, feed_dict):
		# embeddings
		FM_embedding, linear_value = self._get_embeddings_FM(feed_dict)
        # FM_embedding: [bsz, n_item, n_feat, emb_dim]
		bsz, n_item, n_feat, emb_dim = FM_embedding.shape
		# graph
		h_list = self.AutoGraph(FM_embedding)
		y_pred=0.
		for h, fsw in zip(h_list, self.final_score_weight):
			embs_list = self.ep_ip_flatten_emb(h)
			WX_list = self.all_product(embs_list)
			for WX in WX_list:
				y_pred += WX
			if not self.only_use_last_layer:
				y_pred = y_pred * fsw
		predictions = y_pred.view(bsz, n_item)
		return {'prediction':predictions}

class AdaGINCTR(ContextCTRModel, AdaGINBase):
	reader, runner = 'ContextReader', 'CTRRunner'
	extra_log_args = ['emb_size','fi_hidden_units','w_hidden_units','num_gnn_layers','loss_n']
	
	@staticmethod
	def parse_model_args(parser):
		parser = AdaGINBase.parse_model_args_AdaGIN(parser)
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self._define_init(args,corpus)
	
	def forward(self, feed_dict):
		out_dict = AdaGINBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict	

class AdaGINTopK(ContextModel,AdaGINBase):
	reader, runner = 'ContextReader', 'BaseRunner'
	extra_log_args = ['emb_size','fi_hidden_units','w_hidden_units','num_gnn_layers','loss_n']
	
	@staticmethod
	def parse_model_args(parser):
		parser = AdaGINBase.parse_model_args_AdaGIN(parser)
		return ContextModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextModel.__init__(self, args, corpus)
		self._define_init(args,corpus)
	
	def forward(self, feed_dict):
		return AdaGINBase.forward(self, feed_dict)


class AutoGraph_Layer(nn.Module):
	def __init__(self, num_fields, embedding_dim, 
				warm_dim, cold_dim, warm_tau=1.0, cold_tau=0.01, 
				only_use_last_layer=1, gnn_layers=3):
		super(AutoGraph_Layer, self).__init__()
		self.num_fields = num_fields
		self.embedding_dim = embedding_dim
		self.gnn_layers = gnn_layers
		self.only_use_last_layer = only_use_last_layer
		self.warm_dim = warm_dim
		self.cold_dim = cold_dim
		self.warm_tau = warm_tau
		self.cold_tau = cold_tau
		self.all_node_index = nn.Parameter(
			torch.triu_indices(self.num_fields, self.num_fields, offset=-(self.num_fields - 1)), requires_grad=False)
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
		self.warm_W_transform = nn.Sequential(nn.Linear(self.embedding_dim, self.warm_dim),
												nn.ReLU(),
												nn.Linear(self.warm_dim, 1, bias=False))
		self.cold_W_transform = nn.Linear(self.embedding_dim * 2, 1, bias=False)
		self.W_GraphSage = torch.nn.Parameter(torch.Tensor(self.num_fields, self.embedding_dim, self.embedding_dim),
												requires_grad=True)
		nn.init.xavier_normal_(self.W_GraphSage)
		self.eye_mask = nn.Parameter(torch.eye(self.num_fields).bool(), requires_grad=False)

	def build_warm_matrix(self, feature_emb, warm_tau):
		transformer = self.warm_W_transform(feature_emb)
		warm_matrix = F.gumbel_softmax(transformer, tau=warm_tau, dim=1)
		return warm_matrix

	def build_cold_matrix(self, feature_emb, cold_tau):  # Sparse adjacency matrix
		emb1 = torch.index_select(feature_emb, -2, self.all_node_index[0])
		emb2 = torch.index_select(feature_emb, -2, self.all_node_index[1])
		concat_emb = torch.cat((emb1, emb2), dim=-1)
		alpha = self.leaky_relu(self.cold_W_transform(concat_emb))
		alpha = alpha.view(-1, self.num_fields, self.num_fields) # bsz, n_item, n_feat, n_feat
		cold_matrix = F.gumbel_softmax(alpha, tau=cold_tau, dim=-1)
		mask = cold_matrix.gt(0)
		cold_matrix_allone = torch.masked_fill(cold_matrix, mask, float(1))
		cold_matrix_allone = torch.masked_fill(cold_matrix_allone, self.eye_mask, float(1))
		return cold_matrix_allone # bsz, n_item, n_feat, n_feat

	def forward(self, feature_emb):
		bsz, n_item, n_feat, emb_dim = feature_emb.shape
		feature_emb = feature_emb.view(bsz*n_item, n_feat, emb_dim)
		h = feature_emb
		h_list = []
		for i in range(self.gnn_layers):
			cold_matrix = self.build_cold_matrix(h, self.cold_tau)
			new_feature_emb = torch.bmm(cold_matrix, h)
			new_feature_emb = torch.matmul(self.W_GraphSage, new_feature_emb.unsqueeze(-1)).squeeze(-1)
			warm_matrix = self.build_warm_matrix(new_feature_emb, self.warm_tau)
			new_feature_emb = new_feature_emb * warm_matrix
			new_feature_emb = self.leaky_relu(new_feature_emb)
			if not self.only_use_last_layer:
				h_list.append(h)
			else:
				if self.gnn_layers == (i + 1):
					h_list.append(h)
			h = new_feature_emb + feature_emb  # ResNet
		return h_list
