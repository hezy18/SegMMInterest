
"""
Reference:
	'Deep interest evolution network for click-through rate prediction', Zhou et al., AAAI2019.
Implementation reference: DIEN and FuxiCTR
	https://github.com/mouna99/dien/
	https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DIEN/src/DIEN.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from models.context_seq.ClipDINRec import ClipDINRecBase
from utils.layers import MLP_Block
import torch.nn.functional as F

class ClipDIENRecBase(ClipDINRecBase):
	@staticmethod
	def parse_model_args_ClipDIEN(parser):
		parser.add_argument('--emb_size',type=int, default=64, 
							help='the size of the embedding vectors')
		parser.add_argument('--evolving_gru_type',type=str, default='AGRU', 
							help='the type of the evolving gru, select from: AGRU, AUGRU, AIGRU')
		parser.add_argument('--fcn_hidden_layers',type=str, default='[64]',
							help='the hidden layers of the fcn net')
		parser.add_argument('--fcn_activations',type=str,default='ReLU',
							help='the activation function of the fcn net')
		parser.add_argument('--aux_hidden_layers',type=str, default='[64]',
							help='the hidden layers of the auxiliary net')
		parser.add_argument('--aux_activations',type=str,default='ReLU',
							help='the activation function of the auxiliary net')
		parser.add_argument('--alpha_aux',type=float,default=0,
							help='the weight of auxiliary loss, auxiliary net is used only when alpha_aux>0')
		parser.add_argument('--adjust_interest_weight', type=int, default=0,
							help="Whether to adjust interest weight or not.")
		parser.add_argument('--duration_mask', type=int, default=0,
							help="Whether to get duration mask or not.")
		parser.add_argument('--contrastive', type=str, default='',
                            help='Choose the contrastive loss if used')
		parser.add_argument('--norm_interest_type', type=str, default='none', choices=['softmax', 'sigmoid','none'], 
							help="Which normalized type to use on interest logits.") # original:softmax_interest,but now Do on logit
		return parser

	def _define_init_ClipDien(self, args,corpus):
		# DIEN
		self.vec_size = args.emb_size
		self.evolving_gru_type = args.evolving_gru_type
		self.alpha_aux = args.alpha_aux

		self.fcn_hidden_layers = eval(args.fcn_hidden_layers)
		self.fcn_activations = args.fcn_activations

		self.aux_hidden_layers = eval(args.aux_hidden_layers)
		self.aux_activations = args.aux_activations

		self.user_context = ['user_id']+corpus.user_feature_names
		self.item_context = ['item_id']+corpus.item_feature_names
		self.situation_context = corpus.situation_feature_names
		self.gru_emb_size = self.vec_size * (len(self.item_context)+self.add_historical_situations*len(self.situation_context))
		self.fcn_embedding_size = self.vec_size*(len(self.user_context)+len(self.situation_context)+len(self.item_context))+\
						self.gru_emb_size*3

		# clip
		self.contrastive = args.contrastive
		self.adjust_interest_weight = args.adjust_interest_weight
		if self.adjust_interest_weight:
			self.trainable_interest_weight = torch.nn.Parameter(torch.ones(40), requires_grad=True) # 这里是对所有交互固定的clip概率，而不是f(user item time)
		self.duration_mask = args.duration_mask
		self.norm_interest_type = args.norm_interest_type

		self._define_params_DIEN()
		self.apply(self.init_weights)

	def _define_params_DIEN(self):
		# for embedding:
		self.user_embedding = nn.Embedding(self.feature_max['user_id'], self.vec_size)
		self.item_embedding = nn.Embedding(self.feature_max['item_id'], self.vec_size)
		self.item_feature_embedding = nn.Linear(len(self.item_context)-1, self.vec_size)	

		self.frame_position_embedding = nn.Linear(1, self.vec_size)
		if len(self.clip_feature_path):
			self.frame_feature_dim=1024
			self.frame_embedding = nn.Linear(self.frame_feature_dim, self.vec_size)
		self.frame_id_projector = nn.Linear(self.vec_size*2, self.vec_size)
		
		# DIEN
		self.gru = nn.GRU(input_size=self.gru_emb_size, hidden_size=self.gru_emb_size,batch_first=True)

		self.attentionW = nn.Parameter(torch.randn(self.gru_emb_size,self.gru_emb_size),requires_grad=True)

		if self.evolving_gru_type in ["AGRU", "AUGRU"]:
			self.evolving_gru = DynamicGRU(self.gru_emb_size, self.gru_emb_size,
								  gru_type=self.evolving_gru_type)
		elif self.evolving_gru_type == 'AIGRU':
			self.evolving_gru = nn.GRU(input_size=self.gru_emb_size, hidden_size=self.gru_emb_size,
							  		batch_first=True)
		else:
			raise ValueError("Unexpected type of evolving gru: %s"%(self.evolving_gru_type))
		self.fcn_net = MLP_Block(input_dim=self.fcn_embedding_size, output_dim=1,
						   hidden_units=self.fcn_hidden_layers, hidden_activations=self.fcn_activations,
						   output_activation=None, dropout_rates=self.dropout)
		if self.alpha_aux>0: # define aux net and loss
			self.aux_net = MLP_Block(input_dim=self.gru_emb_size*2, output_dim=1,
								hidden_units=self.aux_hidden_layers, hidden_activations=self.aux_activations,
							 output_activation='Sigmoid',dropout_rates=self.dropout)
			self.aux_fn = nn.BCELoss(reduction='none')

	def get_all_embeddings(self, feed_dict):
		history_emb, target_emb, item_feats_emb, user_feats_emb = ClipDINRecBase._get_all_embedding_ClipDIN(self, feed_dict, merge_all=False)
		# history_emb, target_emb, user_feats_emb, situ_feats_emb = DINBase.get_all_embedding(self, feed_dict, merge_all=False)
		return target_emb, history_emb, user_feats_emb, None
		# return target_emb, history_emb, user_feats_emb.flatten(start_dim=-2), situ_feats_emb.flatten(start_dim=-2)\
	  	# 			if not self.add_historical_situations else None

	def get_neg_embeddings(self, feed_dict):
		history_item_id_emb = self.item_embedding(feed_dict['history_neg_item_id'])
		if 'i_duration' in feed_dict:
			history_feature_emb = self.item_feature_embedding(feed_dict['history_neg_i_duration'].float().unsqueeze(-1))
		history_item_emb = torch.cat([history_item_id_emb, history_feature_emb], dim=-1)
		# history_item_feat_emb = torch.stack([self.embedding_dict[f](feed_dict['history_neg_'+f]) if f.endswith('_c') or f.endswith('_id')
		# 			else self.embedding_dict['history_'+f](feed_dict[f].float().unsqueeze(-1))
		# 			for f in self.item_context],dim=-2) # batch * feature num * emb size
		if self.add_historical_situations:
			# TODO:
			history_situ_emb = torch.stack([self.embedding_dict[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.situation_context],dim=-2) # batch * feature num * emb size
			history_emb = torch.cat([history_item_emb,history_situ_emb],dim=-2).flatten(start_dim=-2)
		else:
			history_emb = history_item_emb.flatten(start_dim=-2)
		return history_emb # batch , (feature num * emb size)

	def interest_extractor(self, history_emb, hislens):
		history_packed = nn.utils.rnn.pack_padded_sequence(
			history_emb, hislens.cpu(), batch_first=True, enforce_sorted=False)
		output, hidden = self.gru(history_packed, None)
		
		interest_emb, _ = nn.utils.rnn.pad_packed_sequence(
				output, batch_first=True,total_length=history_emb.shape[1])
		return output, interest_emb 

	def target_attention(self, target_emb, interest_emb, lengths):
		# target_emb: batch * item num * emb size; interest_emb: batch * his len * emb size
		batch_size = target_emb.shape[0]
		attention_prod = torch.bmm(interest_emb, self.attentionW.unsqueeze(dim=0).repeat(batch_size, 1,1)) # batch * len * emb size
		attention_prod = (attention_prod * target_emb.unsqueeze(dim=1)).sum(dim=-1) # (batch * item num) * his len
		attention_score = (attention_prod-attention_prod.max()).softmax(dim=-2)
		return attention_score # (batch*item num) * his len

	def interest_evolving(self, interests, interest_emb, target_emb, lengths):
		attention = self.target_attention(target_emb, interest_emb, lengths) # (batch * item num) * his len
		if self.evolving_gru_type == 'AIGRU':
			packed_inputs = nn.utils.rnn.pack_padded_sequence(interest_emb*attention.unsqueeze(dim=-1),
									lengths = lengths.cpu(), batch_first=True, enforce_sorted=False)
			_, hidden_out = self.evolving_gru(packed_inputs)
		else:
			packed_scores = nn.utils.rnn.pack_padded_sequence(attention,
									lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)
			_, hidden_out = self.evolving_gru(interests, packed_scores)
		return hidden_out

	def _get_inp(self, feed_dict):
		hislens = feed_dict['lengths']
		# target_emb:[256, 1, 40, 64]; history_emb:[256, 20, 64]; user_emb(context_emb):[256, 1, 40, 128]
		target_emb, history_emb, user_emb, context_emb = self.get_all_embeddings(feed_dict)
		batch_size, item_num, clip_num, _ = target_emb.shape
		history_sum = history_emb.sum(dim=-2) # [256,64]
		
		# pass this line first(先只考虑alpha_aux=0的情况)
		neg_history_emb = self.get_neg_embeddings(feed_dict) if self.alpha_aux>0 and feed_dict['phase']=='train' else None

		# repeat and flatten history according to multiple targets
		history_emb_2d = history_emb.unsqueeze(dim=1).repeat(1,item_num,clip_num,1,1).reshape(-1, history_emb.shape[-2], history_emb.shape[-1])
		hislens_2d = hislens.unsqueeze(dim=1).repeat(1,item_num,clip_num).reshape(-1)
		target_emb_2d = target_emb.reshape(-1, target_emb.shape[-1])
		# target_emb_2d:[10240, 64]; history_emb:[10240, 20, 64]; hislens_2d:[10240]
		gru_output, interest_emb = self.interest_extractor(history_emb_2d,hislens_2d)
		h_out = self.interest_evolving(gru_output, interest_emb, target_emb_2d, hislens_2d).reshape(batch_size, item_num, clip_num, -1)
		
		if context_emb is None:
			inp = torch.cat([user_emb.unsqueeze(1).unsqueeze(1).repeat(1,item_num,clip_num,1), 
							target_emb, 
							history_sum.unsqueeze(1).unsqueeze(1).repeat(1,item_num,clip_num, 1), 
					  		target_emb*history_sum.unsqueeze(1).unsqueeze(1), h_out],
				  			dim=-1)
		else:
			inp = torch.cat([user_emb.unsqueeze(dim=1).repeat(1,item_num,1), 
				   		context_emb.unsqueeze(dim=1).repeat(1,item_num,1), 
					 	target_emb, history_sum.unsqueeze(dim=1).repeat(1,item_num,1), 
					  	target_emb*history_sum.unsqueeze(dim=1), h_out],
				  		dim=-1)
		
		if self.alpha_aux>0 and feed_dict['phase']=='train':
			interest_emb = interest_emb.reshape(batch_size, item_num, history_emb.size(1),-1)[:,0,:,:].squeeze(dim=1) 
			interest_emb = torch.sum(interest_emb.reshape(batch_size, history_emb.size(1), clip_num,-1), dim=-2)
			neg_history_emb = neg_history_emb.reshape(batch_size, history_emb.size(1), -1)
			return inp, {'neg_history':neg_history_emb, 'pos_history':history_emb,'interest_emb':interest_emb,'lengths':hislens}
		else:
			return inp, {}

	def _clip_integret_Rec_DIEN(self, clip_predictions, feed_dict):
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
		inp, out_dict = self._get_inp(feed_dict)
		# Fully connected layer
		clip_predictions = self.fcn_net(inp).squeeze(dim=-1)

		predictions = self._clip_integret_Rec_DIEN(clip_predictions, feed_dict)

		out_dict['prediction'] = predictions
		return out_dict

	def aux_loss(self, out_dict):
		''' Auxiliary loss
  		Reference: https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DIEN/src/DIEN.py
		'''
		interest_emb, pos_emb, neg_emb, lengths = out_dict["interest_emb"], out_dict["pos_history"], \
												   out_dict["neg_history"], out_dict["lengths"]
		batch_size, his_lens, emb_size = interest_emb.shape
		pos_prob = self.aux_net(torch.cat([interest_emb[:, :-1, :], pos_emb[:, 1:, :]], dim=-1).view(-1, emb_size * 2))
		neg_prob = self.aux_net(torch.cat([interest_emb[:, :-1, :], neg_emb[:, 1:, :]], dim=-1).view(-1, emb_size * 2))
		aux_prob = torch.cat([pos_prob, neg_prob], dim=0).view(-1, 1)
		aux_label = torch.cat([torch.ones_like(pos_prob, device=aux_prob.device),
								   torch.zeros_like(neg_prob, device=aux_prob.device)], dim=0).view(-1, 1)
		aux_loss = self.aux_fn(aux_prob, aux_label).view(2, batch_size, his_lens-1)
		pad_mask = (torch.arange(his_lens)[None,:].to(self.device)<lengths[:,None])[:,1:]
		pad_mask_aux = torch.stack([pad_mask,pad_mask],dim=0)
		aux_loss = torch.sum(aux_loss * pad_mask_aux, dim=-1) / (torch.sum(pad_mask, dim=-1)+1e-9)
		return aux_loss.mean()

	class Dataset(object):
		def _get_feed_dict_dien(self, index, feed_dict):
			if self.model.alpha_aux>0 and self.phase=='train':
				pos = self.data['position'][index]
				user_neg_seq = self.data['neg_user_his'][feed_dict['user_id']][:pos]
				if self.model.history_max > 0:
					user_neg_seq = user_neg_seq[-self.model.history_max:]
				feed_dict['history_neg_item_id'] = user_neg_seq 
				for c in self.corpus.item_feature_names:
					feed_dict['history_neg_'+c] = np.array([self.corpus.item_features[iid][c] if iid in self.corpus.item_features else 0
                                             for iid in feed_dict['history_neg_item_id']])
				
			return feed_dict

		def actions_before_epoch_dien(self):
			if self.model.alpha_aux>0:
				neg_history = dict() # user: negative history
				for u, his in self.corpus.user_his.items():
					neg_history[u] = np.random.randint(1, self.corpus.n_items,
										size=(len(his)))
					for i,(pos, neg) in enumerate(zip(his, neg_history[u])):
						while pos[0] == neg:
							neg = np.random.randint(1, self.corpus.n_items)
						neg_history[u][i] = neg
				self.data['neg_user_his'] = neg_history	

class ClipDIENRecRanking(ContextSeqModel, ClipDIENRecBase):
	reader, runner = 'ContextSeqReader', 'BaseRunner'
	extra_log_args = ['emb_size','evolving_gru_type','fcn_hidden_layers','aux_hidden_layers','alpha_aux','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']
	
	@staticmethod
	def parse_model_args(parser):
		parser = ClipDIENRecBase.parse_model_args_ClipDIEN(parser)
		return ContextSeqModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init_ClipDien(args,corpus)

	def forward(self, feed_dict):
		return ClipDIENRecBase.forward(self, feed_dict)

	def loss(self, out_dict):
		loss = ContextSeqModel.loss(self, out_dict)
		if self.alpha_aux>0:
			loss += self.alpha_aux*self.aux_loss(out_dict)
		return loss

	class Dataset(ContextSeqModel.Dataset, ClipDIENRecBase.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = ContextSeqModel.Dataset._get_feed_dict(self, index)
			feed_dict = ClipDIENRecBase.Dataset._get_feed_dict_dien(self, index, feed_dict)
			return feed_dict

		def actions_before_epoch(self):
			super().actions_before_epoch()
			ContextSeqModel.Dataset.actions_before_epoch(self)
			DIENBase.Dataset.actions_before_epoch_dien(self)

class ClipDIENRecCTR(ContextSeqCTRModel, ClipDIENRecBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size','evolving_gru_type','fcn_hidden_layers','aux_hidden_layers','alpha_aux','adjust_interest_weight','clip_weight_path','auxillary_loss_weight']

	@staticmethod
	def parse_model_args(parser):
		parser = ClipDIENRecBase.parse_model_args_ClipDIEN(parser)
		return ContextSeqCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init_ClipDien(args,corpus)

	def forward(self, feed_dict):
		out_dict = ClipDIENRecBase.forward(self,feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

	def loss(self, out_dict):
		loss = ContextSeqCTRModel.loss(self, out_dict)
		if self.alpha_aux>0:
			loss += self.alpha_aux*self.aux_loss(out_dict)
		return loss

	class Dataset(ContextSeqCTRModel.Dataset, ClipDIENRecBase.Dataset):
		def _get_feed_dict(self, index):
			feed_dict = ContextSeqCTRModel.Dataset._get_feed_dict(self, index)
			feed_dict = ClipDIENRecBase.Dataset._get_feed_dict_dien(self, index, feed_dict)
			return feed_dict

		def actions_before_epoch(self):
			ContextSeqCTRModel.Dataset.actions_before_epoch(self)
			ClipDIENRecBase.Dataset.actions_before_epoch_dien(self)


class DynamicGRU(nn.Module):
	"""DynamicGRU with GRU, AIGRU, AGRU, and AUGRU choices
		Reference: https://github.com/GitHub-HongweiZhang/prediction-flow/blob/master/prediction_flow/pytorch/nn/rnn.py
	"""
	def __init__(self, input_size, hidden_size, bias=True, gru_type='AUGRU'):
		super(DynamicGRU, self).__init__()
		self.hidden_size = hidden_size
		self.gru_type = gru_type
		if gru_type == "AUGRU":
			self.gru_cell = AUGRUCell(input_size, hidden_size, bias=bias)
		elif gru_type == "AGRU":
			self.gru_cell = AUGRUCell(input_size, hidden_size, bias=bias)
	
	def forward(self, packed_seq_emb, attn_score=None, h=None):
		assert isinstance(packed_seq_emb, nn.utils.rnn.PackedSequence) and isinstance(
					attn_score, nn.utils.rnn.PackedSequence), \
			   "DynamicGRU supports only `PackedSequence` input."
		x, batch_sizes, sorted_indices, unsorted_indices = packed_seq_emb
		attn, _, _, _ = attn_score
		
		if h == None:
			h = torch.zeros(batch_sizes[0], self.hidden_size, device=x.device)
		output_h = torch.zeros(batch_sizes[0], self.hidden_size, device=x.device)
		outputs = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
		
		start = 0
		for batch_size in batch_sizes:
			_x = x[start: start + batch_size]
			_h = h[:batch_size]
			_attn = attn[start: start + batch_size]
			h = self.gru_cell(_x, _h, _attn)
			outputs[start: start + batch_size] = h
			output_h[:batch_size] = h
			start += batch_size
		
		return nn.utils.rnn.PackedSequence(outputs, batch_sizes, sorted_indices, unsorted_indices), \
			   output_h[unsorted_indices]
			   
class AUGRUCell(nn.Module):
	"""AUGRUCell with attentional update gate
		Reference1: https://github.com/reczoo/FuxiCTR/blob/v2.0.1/model_zoo/DIEN/src/DIEN.py#L287
		Reference2: GRUCell from https://github.com/emadRad/lstm-gru-pytorch/blob/master/lstm_gru.ipynb
	"""
	def __init__(self, input_size, hidden_size, bias=True):
		super(AUGRUCell, self).__init__()
		self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
		self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

	def forward(self, x, hx, attn):
		gate_x = self.x2h(x) 
		gate_h = self.h2h(hx)
		
		i_u, i_r, i_n = gate_x.chunk(3, 1)
		h_u, h_r, h_n = gate_h.chunk(3, 1)
		
		update_gate = torch.sigmoid(i_u + h_u)
		update_gate = update_gate * attn.unsqueeze(-1)
		reset_gate = torch.sigmoid(i_r + h_r)
		new_gate = torch.tanh(i_n + reset_gate * h_n)
		hy = hx + update_gate * (new_gate - hx)
		return hy
