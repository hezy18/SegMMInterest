import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, repeat, rearrange, reduce
#from kn_util.basic import registry
from model.ms_temporal_detr.ms_pooler import MultiScaleRoIAlign1D
from misc import cw2se, calc_iou
from .loss import l1_loss, iou_loss
from kn_util.nn_utils.layers import MLP
from kn_util.nn_utils import clones
from kn_util.nn_utils.math import inverse_sigmoid_torch, gaussian_torch
#from kn_util.basic import registry, global_get
from torchvision.ops import sigmoid_focal_loss
from kn_util.nn_utils.init import init_module
import copy
import os
from kn_util.basic import eval_env
from typing import Optional, Any, Union, Callable
from torch import Tensor
from torch.nn import Linear, MultiheadAttention, Dropout, LayerNorm
from torch.nn.modules.transformer import _get_activation_fn
from kn_util.basic import eval_env
import os.path as osp
import time
import os


GROUP_IDX = eval_env("KN_DENOISING", 1)
KN_QUERY_ATTN = eval_env("KN_QUERY_ATTN", False)
KN_QUERY_ATTN_NOSHARE = eval_env("KN_QUERY_ATTN_NOSHARE", False)
KN_P_ADD_T = eval_env("KN_P_ADD_T", False)
local_dict = copy.copy(locals())
print({k: v for k, v in local_dict.items() if k.startswith("KN")})

def my_sigmoid_focal_loss(inputs, targets, alpha, gamma, reduction="none", exposure_prob=None):
    p = torch.sigmoid(inputs)
    # correction with exposure prob
    exposure_prob = repeat(torch.tensor(exposure_prob, device=targets.device), "L -> b L", b=inputs.shape[0])
    p = p * exposure_prob
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss

def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    huber = torch.where(torch.abs(error) < delta,
                       0.5 * torch.pow(error, 2),
                       delta * (torch.abs(error) - 0.5 * delta))
    return torch.mean(huber)

def compute_leave_prob_CE(h_t, y, mask=None, reduction="mean"):
    # loss = F.nll_loss(h_t, y)
    p = h_t
    exp_p = torch.exp(p)
    ce_loss = F.binary_cross_entropy_with_logits(exp_p, y, reduction="none")
    # ce_loss = -(y * p + (1-y) * torch.log(1 - exp_p + 1e-12))


    # import pdb; pdb.set_trace()
    # # for focal loss
    # p_t = exp_p * y + (1 - exp_p) * (1 - y)
    # gamma, alpha = 2, 0.5
    # loss = ce_loss * ((1 - p_t) ** gamma)
    # if alpha >= 0:
    #     alpha_t = alpha * y + (1 - alpha) * (1 - y)
    #     loss = alpha_t * loss
    loss = ce_loss
    masked_loss = loss * mask
    if reduction == "none":
        loss = masked_loss
    elif reduction == "mean":
        loss = masked_loss.sum()/mask.sum()
    elif reduction == "sum":
        pass
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    # import pdb;pdb.set_trace()
    return loss

def compute_interest_leave_CE(logits, gt, mask=None, loss='CE', use_mask = 0, reduction="mean"):
    # loss = F.nll_loss(h_t, y)
    gt_nonleave = (gt != 0).float()
    norm_interest = logits.softmax(dim=1)
    norm_gt = gt_nonleave.softmax(dim=1)
    # print(norm_interest)
    # print(norm_gt)
    # import pdb; pdb.set_trace()
    # ce_loss = F.binary_cross_entropy_with_logits(norm_interest, norm_gt,reduction="none") # 如果要改成focal loss，就需要reduction
    if loss=='CE':
        if use_mask:
            # import pdb; pdb.set_trace()
            mask_ce_loss =  -torch.sum(mask * norm_gt * norm_interest.log(), dim=1)
            ce_loss = mask_ce_loss / mask.sum(dim=1)
            ce_loss = ce_loss.mean()
            # mask_kl_loss = F.kl_div(norm_interest.log(), norm_gt, reduction="none") * mask
            # kl_loss = mask_kl_loss.sum(dim=1) / mask.sum(dim=1)
            # kl_loss = kl_loss.mean()
        else:
            ce_loss = -torch.sum(norm_gt * norm_interest.log(), dim=1).mean()
        loss = ce_loss
    elif loss=='KL':
        if use_mask:
            mask_kl_loss = F.kl_div(norm_interest.log(), norm_gt, reduction="none") * mask
            kl_loss = mask_kl_loss.sum(dim=1) / mask.sum(dim=1)
            kl_loss = kl_loss.mean()
        else:
            kl_loss = F.kl_div(norm_interest.log(), norm_gt, reduction="batchmean")
        loss = kl_loss
    # print(norm_gt)
    # print(norm_interest)
    # print(ce_loss)
    # import pdb; pdb.set_trace()

    # ce_loss = F.binary_cross_entropy_with_logits(p, y, reduction="none")
    # ce_loss = -(y * p + (1-y) * torch.log(1 - exp_p + 1e-12))

    # p_t = exp_p * y + (1 - exp_p) * (1 - y)
    # gamma, alpha = 2, 0.5
    # loss = ce_loss * ((1 - p_t) ** gamma)
    
    # if alpha >= 0:
    #     alpha_t = alpha * y + (1 - alpha) * (1 - y)
    #     loss = alpha_t * loss
    # masked_loss = ce_loss * mask
    # print('ce_loss=',ce_loss, '    kl_loss=',kl_loss)

    # loss = ce_loss # mask么？是不是要在softmax前先mask？但是如何批量地在softmax前mask呢
    # import pdb; pdb.set_trace()
    # for focal loss
    # p_t = p * targets + (1 - p) * (1 - targets) 
    # loss = ce_loss * ((1 - p_t) ** gamma)
    # if reduction == "none":
    #     loss = masked_loss
    # elif reduction == "mean":
    #     loss = masked_loss.sum()/mask.sum()
    # elif reduction == "sum":
    #     pass
    # else:
    #     raise ValueError(
    #         f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
    #     )
    return loss

def compute_interest_BPR_all(leave_probs, view_lengths,mask):
    """
    BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
    "Recurrent neural networks with top-k gains for session-based recommendations"
    :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
    :return:
    """
    bsz, seq_len = leave_probs.shape
    view_lengths = view_lengths.to(torch.int64)
    valid_row_mask = ((view_lengths < 40)).squeeze(-1)
    # mask
    # valid_row_mask = (view_lengths!=mask_batch.sum(axis=1))
    bsz = valid_row_mask.sum().item()
    view_lengths = view_lengths[valid_row_mask,:]
    leave_probs = leave_probs[valid_row_mask,:]
    mask = mask[valid_row_mask,:]
    
    indices = torch.arange(leave_probs.shape[1])

    view_lengths = view_lengths.view(-1)

    # import pdb; pdb.set_trace()
    pos_pred = leave_probs[torch.arange(leave_probs.size(0)), view_lengths]

    neg_mask = torch.ones_like(leave_probs, dtype=torch.bool)

    neg_mask[torch.arange(leave_probs.size(0)), view_lengths] = False

    neg_pred = leave_probs[neg_mask]

    neg_pred = neg_pred.view(-1, seq_len-1)


    # batch_indices = torch.arange(batch_size).unsqueeze(1)
    # pos_pred = leave_prob.gather(1, view_lengths.unsqueeze(1))
    # mask = torch.ones_like(leave_prob, dtype=torch.bool)
    # mask[batch_indices, view_lengths.unsqueeze(1)] = 0 

    # neg_pred = leave_prob[mask].view(batch_size, seq_len - 1)

    # print("pos_pred:", pos_pred)
    # print("neg_pred:", neg_pred)
    # pos_softmax = (pos_pred - pos_pred.max()).softmax(dim=1)


    # print('pos_pred: ', pos_pred.min(), pos_pred.max())
    # print('neg_pred: ',neg_pred.min(), neg_pred.max())

    # print('pos_pred : ', pos_pred[:10])
    # print('leave_probs : ',leave_probs)
    neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
    neg_softmax2 = (neg_pred.min() - neg_pred).softmax(dim=1)
    soft_diff = (neg_pred - pos_pred[:, None]).sigmoid() * neg_softmax
    loss = -(soft_diff.sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
    # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
    # loss = F.softplus(-(pos_pred - neg_pred)).mean()
    # ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
    # import pdb; pdb.set_trace()
    return loss

def compute_interest_BPR(leave_probs, view_lengths,mask):
    """
    BPR ranking loss with optimization on multiple negative samples (a little different now to follow the paper ↓)
    "Recurrent neural networks with top-k gains for session-based recommendations"
    :param out_dict: contain prediction with [batch_size, -1], the first column for positive, the rest for negative
    :return:
    """
    bsz, seq_len = leave_probs.shape
    view_lengths = view_lengths.to(torch.int64)
    valid_row_mask = ((view_lengths < 40) & (view_lengths > 0)).squeeze(-1)
    # valid_row_mask = (view_lengths < 40).squeeze(-1)
    bsz = valid_row_mask.sum().item()
    view_lengths = view_lengths[valid_row_mask,:]
    leave_probs = leave_probs[valid_row_mask,:]
    
    indices1 = view_lengths.expand(-1, leave_probs.size(1))
    # print(leave_probs.shape,torch.max(leave_probs),torch.min(leave_probs))
    # import pdb; pdb.set_trace()
    # print(indices1.shape,torch.max(indices1),torch.min(indices1))
    pos_pred = torch.gather(leave_probs, 1, indices1)[:,0]

    indices2 = torch.arange(seq_len).expand(bsz, -1).to(view_lengths.device)
    # view_lengths[view_lengths==0]=1
    mask = indices2 < view_lengths
    fill_values = pos_pred.unsqueeze(1).expand_as(leave_probs)
    # fill_values = leave_probs[:, :1].expand_as(mask)
    neg_pred = torch.where(mask, leave_probs, fill_values)
    

    # print('leave_probs: ',leave_probs.min(), leave_probs.max())
    # print('pos_pred: ', pos_pred.min(), pos_pred.max())
    # print('neg_pred: ',neg_pred.min(), neg_pred.max())

    # print('pos_pred : ', pos_pred[:10])
    # print('leave_probs : ',leave_probs)
    # import pdb; pdb.set_trace()

    neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1) 
    soft_diff = (neg_pred - pos_pred[:, None]).sigmoid() * neg_softmax
    soft_diff_zero =  torch.where(mask, soft_diff,0)

    # loss = -(soft_diff.sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
    loss = -(soft_diff_zero.sum(dim=1)/mask.sum(dim=1)).clamp(min=1e-8,max=1-1e-8).log().mean()
    # neg_pred = (neg_pred * neg_softmax).sum(dim=1)
    # loss = F.softplus(-(pos_pred - neg_pred)).mean()
    # ↑ For numerical stability, use 'softplus(-x)' instead of '-log_sigmoid(x)'
    # import pdb; pdb.set_trace()
    return loss


def compute_partial_likelihood_loss(hazard, y, mask=None):
    log_likelihood = 0
    n_samples = y.size(0)

    for i in range(n_samples):
        
        observed_time = y[i].long()  
        if observed_time==40: # TODO hzy：面临观看时间=duration的问题（这里只处理了40，以防止报错）
            continue
        # import pdb; pdb.set_trace()
        log_hazard = torch.log(hazard[i, observed_time][0] + 1e-6)  # log P(y_i)
        risk_set = torch.sum(hazard[i, observed_time:])  # 风险集总和
        log_likelihood += log_hazard - torch.log(risk_set + 1e-6)
    return -log_likelihood / n_samples  # 返回负对数似然损失
        

class TransformerDecoderLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model,
                                                 nhead,
                                                 dropout=dropout,
                                                 batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self,
                tgt: Tensor,
                memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        x, attn = self.multihead_attn(x,
                                      mem,
                                      mem,
                                      attn_mask=attn_mask,
                                      key_padding_mask=key_padding_mask,
                                      need_weights=True)

        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class InteractionAggregation(nn.Module):
	def __init__(self, x_dim, y_dim, output_dim=1, num_heads=1):
		super(InteractionAggregation, self).__init__()
		
		self.num_heads = num_heads
		self.output_dim = output_dim
		self.w_x = nn.Linear(x_dim, output_dim)
		self.w_y = nn.Linear(y_dim, output_dim)
		self.w_x.apply(init_module)
		self.w_y.apply(init_module)
		if num_heads>0:
			assert x_dim % num_heads == 0 and y_dim % num_heads == 0, \
                "Input dim must be divisible by num_heads!"
			self.head_x_dim = x_dim // num_heads
			self.head_y_dim = y_dim // num_heads
			self.w_xy = nn.Parameter(torch.Tensor(num_heads * self.head_x_dim * self.head_y_dim, 
												output_dim))
			nn.init.xavier_normal_(self.w_xy)

	def forward(self, x, y):
		batch_size, item_num = x.shape[0], x.shape[1]
		output = self.w_x(x) + self.w_y(y)
		if self.num_heads>0:
			head_x = x.view(batch_size, item_num, self.num_heads, self.head_x_dim).flatten(start_dim=0,end_dim=1)
			head_y = y.view(batch_size, item_num, self.num_heads, self.head_y_dim).flatten(start_dim=0,end_dim=1)
			xy = torch.matmul(torch.matmul(head_x.unsqueeze(2), 
                                        self.w_xy.view(self.num_heads, self.head_x_dim, -1)) \
                                .view(-1, self.num_heads, self.output_dim, self.head_y_dim),
                            head_y.unsqueeze(-1)).squeeze(-1)
			xy_reshape = xy.sum(dim=1).view(batch_size,item_num,-1)
			output += xy_reshape
		return output.squeeze(-1)

class MultiScaleTemporalDetrLeaveFocal(nn.Module):  # first

    def __init__(self, backbone1, backbone2, head,frame_pooler, model_cfg) -> None:
        super().__init__()
        
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.model_cfg = model_cfg
        self.head = head
        self.frame_pooler = frame_pooler

        self.debug = model_cfg.debug
        self.input_type = model_cfg.input_type # e.g., input_type ={'user':'image','photo':'image'}

        d_model = model_cfg.d_model
        self.bias_weight = None
        self.bias_bias = None
        if model_cfg.learnable_bias:
            self.bias_weight = torch.nn.Parameter(torch.ones(1, 40), requires_grad=True)
            self.bias_bias = torch.nn.Parameter(torch.ones(1, 40), requires_grad=True)
        self.exposure_prob = model_cfg.exposure_prob

        
            
        # stage mlp
        if self.backbone2==None:
            if head==None: self.stage_mlp1 = nn.Linear(d_model, 1)
            else:
                self.stage_mlp1 = nn.Linear(d_model, 3 * d_model)
                self.stage_mlps = clones(MLP([d_model, d_model, 1]), 3)
                self.stage_mlps.apply(init_module)

            self.stage_mlp1.apply(init_module)
        else:
            if model_cfg.fusion_heads == -2 or model_cfg.fusion_heads == -3: 
                self.stage_mlp1 = nn.Linear(d_model, 1)
                self.stage_mlp1.apply(init_module)
            elif model_cfg.fusion_heads == -1:
                self.stage_mlp1 = nn.Linear(2*d_model, 1)
                self.stage_mlp1.apply(init_module)
            elif model_cfg.fusion_heads == 0:
                self.stage_mlp1 = nn.Linear(d_model, 1)
                self.stage_mlp1.apply(init_module)
                self.stage_mlp2 = nn.Linear(d_model, 1)
                self.stage_mlp2.apply(init_module)
            else:
                self.fusion_module = InteractionAggregation(d_model,d_model,output_dim=1,num_heads=model_cfg.fusion_heads)
            

        
        #self.text_mlp.apply(init_module)

    def compute_iou_loss(self, proposal, score, gt):
        # calc_iou(gt)
        Nc = proposal.shape[1]
        expanded_gt = repeat(gt, "b i -> (b nc) i", nc=Nc)
        proposal = rearrange(proposal, "b nc i -> (b nc) i", nc=Nc)
        iou_type = "giou" if os.getenv("KN_SOFT_GIOU", False) else "iou"
        ious = calc_iou(proposal, expanded_gt, type=iou_type)
        score_flatten = score.flatten()
        return sigmoid_focal_loss(score_flatten, ious, alpha=0.5, reduction="mean")




    def compute_loss(self, stage_logits=None, gt=None):#, text_logits, word_mask, word_label, gt):
        loss_dict={}
        
        bsz = gt.shape[0]
        mask = gt!=-2
        
        bias = None
        if self.bias_weight!=None: #
            pos = torch.arange(40).float().cuda()
            bias = (pos+1) * self.bias_weight + self.bias_bias
            bias = repeat(bias, "1 L -> b L", b=bsz)
        if self.bias_weight==None:
            logits = stage_logits.squeeze(-1)
        else:
            logits = stage_logits.squeeze(-1) + bias  
        
        p = torch.sigmoid(logits) # (batch_size, seq_len)

        h_i = p
        # import pdb; pdb.set_trace()
        # h_t = torch.cumsum(torch.log(h_i+1e-6), dim=1) 
        h_t= torch.cumsum(torch.log(h_i), dim=1) 
        # 计算生存函数 S(t)
        # survival_prob = torch.cumprod(h_i, dim=1)
        survival_prob = torch.exp(h_t)  
        hazard = 1 - survival_prob  # 离开概率 (batch_size, seq_len)

        # gt[gt>0] = 1.0
        # gt[gt==-1] = 0.0
        gt_binary = (gt == 1).float()
        view_lengths =gt_binary.sum(dim=1, keepdim=True).to(gt_binary.device)
        duration_binary = (gt != -2).float()
        durations = duration_binary.sum(dim=1, keepdim=True).to(duration_binary.device).int()
        hazard_masked = hazard.clone()
        hazard_masked[~mask] = 0
        survival_masked = survival_prob.clone()
        survival_masked[~mask] = 0
        

        loss_dict = dict()
        # import pdb;pdb.set_trace()
        
        for loss in self.model_cfg.loss_type_list:
            if loss == 'focal':
                gt[gt>0] = 1.0
                gt[gt==-1] = 0.0
                span_loss_element_wise = my_sigmoid_focal_loss(logits, gt.float(), alpha=0.5, gamma=2, reduction="none", exposure_prob=self.exposure_prob)
                span_loss = torch.sum(span_loss_element_wise[mask]) / bsz
                loss_dict["focal"] = span_loss
            elif loss=='huber':
                loss_dict["huber"] = huber_loss(torch.sum(hazard_masked, dim=1), view_lengths.float(), delta=1.0)
            elif loss=='hazard':
                loss_dict["hazard"] = compute_partial_likelihood_loss(hazard_masked, view_lengths, mask=mask)
            elif loss == 'surviveCE':
                loss_dict['surviveCE'] = compute_leave_prob_CE(h_t, gt_binary,mask=mask)
            elif loss == 'interestBPR':
                # loss_dict['interestBPR'] = compute_interest_BPR(logits, view_lengths,mask=mask)
                loss_dict['interestBPR'] = compute_interest_BPR_all(logits, view_lengths,mask=mask)
            elif loss == 'interestCE':
                loss_dict['interestCE'] = compute_interest_leave_CE(logits, gt, mask=mask, loss='CE', use_mask=self.model_cfg.mask_loss)
            elif loss == 'interestKL':
                loss_dict['interestKL'] = compute_interest_leave_CE(logits, gt, mask=mask, loss='KL', use_mask=self.model_cfg.mask_loss)
        mse1 = nn.MSELoss()(torch.sum(survival_masked, dim=1), view_lengths.float())
        loss_dict["mse"] = mse1
        for i in range(survival_masked.shape[0]): # TODO!:here
            survival_masked[i,durations[i]-1]=1 
        view_lengths2 = (gt >= 0).sum(dim=1, keepdim=True).to(gt_binary.device)
        mse2 = nn.MSELoss()(torch.sum(survival_masked, dim=1), view_lengths2.float())
        loss_dict["mse2"] = mse2
        # import pdb;pdb.set_trace()
        loss_score = 0.
        for loss in self.model_cfg.loss_type_list:
            if loss == 'huber':
                coef = self.model_cfg.loss_weight['mse']
            else:
                coef = self.model_cfg.loss_weight[loss]
            loss_score+=loss_dict[loss]*coef
        loss_dict["loss"] = loss_score
        # print('mse = ',mse, '\t likelihood = ', partial_likelihood_loss)
        print(loss_dict)
        loss_dict['logits'] = logits
        loss_dict['gt'] = gt
        return loss_dict

    def forward(self, usr_image, usr_id, usr_mask, vid_image, vid_id, vid_mask, gt=None, mode="train", **kwargs):
        
        B = usr_id.shape[0]

        model_cfg = self.model_cfg
        vid_image = self.frame_pooler(vid_image)
        usr_image = self.frame_pooler(usr_image)
        # import pdb; pdb.set_trace()
        if self.debug:
            st = time.time()
        if self.backbone2==None:
            if self.input_type['user']=='image':
                usr_feat = usr_image
            elif self.input_type['user']=='id':
                usr_feat = usr_id
            if self.input_type['photo']=='image':
                vid_feat = vid_image
            elif self.input_type['photo']=='id':
                vid_feat = vid_id
            vid_feat_lvls, usr_feat = self.backbone1(usr_feat=usr_feat, usr_mask=usr_mask,
                                                            vid_feat=vid_feat, vid_mask=vid_mask,
                                                            )
            vid_feat_stage = self.stage_mlp1(vid_feat_lvls[-1])
        else:
            if self.input_type['user']=='both':
                usr_feat1 = usr_image
                usr_feat2 = usr_id
            elif self.input_type['user']=='id':
                usr_feat1 = usr_id
                usr_feat2 = usr_id
            elif self.input_type['user']=='image':
                usr_feat1 = usr_image
                usr_feat2 = usr_image
            if self.input_type['photo']=='both':
                vid_feat1 = vid_image
                vid_feat2 = vid_id
            elif self.input_type['photo']=='id':
                vid_feat1 = vid_id
                vid_feat2 = vid_id
            elif self.input_type['photo']=='image':
                vid_feat1 = vid_image
                vid_feat2 = vid_image
            vid_feat_lvls1, usr_feat1 = self.backbone1(usr_feat=usr_feat1, usr_mask=usr_mask,
                                                    vid_feat=vid_feat1, vid_mask=vid_mask,
                                                    ) 
            vid_feat_lvls2, usr_feat2 = self.backbone2(usr_feat=usr_feat2, usr_mask=usr_mask,
                                                    vid_feat=vid_feat2, vid_mask=vid_mask,
                                                    ) 
            # vid_feat_lvls = vid_feat_lvls1 + vid_feat_lvls2
            # import pdb; pdb.set_trace()
            if self.model_cfg.fusion_heads == -3:
                vid_feat_lvls = vid_feat_lvls1 + vid_feat_lvls2
                vid_feat_stage = self.stage_mlp1(vid_feat_lvls[-1])
            elif self.model_cfg.fusion_heads == -2:
                vid_feat_lvls = vid_feat_lvls1[-1] + vid_feat_lvls2[-1]
                vid_feat_stage = self.stage_mlp1(vid_feat_lvls)
            elif self.model_cfg.fusion_heads ==-1:
                vid_feat_lvls = torch.cat([vid_feat_lvls1[-1], vid_feat_lvls2[-1]],dim=-1)
                vid_feat_stage = self.stage_mlp1(vid_feat_lvls)
            elif self.model_cfg.fusion_heads ==0:
                vid_feat_stage = self.stage_mlp1(vid_feat_lvls1[-1]) + self.stage_mlp2(vid_feat_lvls2[-1])
            else:
                vid_feat_stage =  self.fusion_module(vid_feat_lvls1[-1], vid_feat_lvls2[-1])
            
        if self.debug:
            st = time.time()
       
        if self.debug:
            et = time.time()
            print(f"stage mlp time cost: {et-st}")
        # import pdb; pdb.set_trace()        
        if mode in ("train", "test"):
            loss_dict = self.compute_loss(stage_logits=vid_feat_stage, gt=gt)
            return loss_dict
        elif mode == "inference":
            bias = None
            if self.bias_weight!=None:
                pos = torch.arange(40).float().cuda()
                bias = (pos+1) * self.bias_weight + self.bias_bias
                bsz = usr_image.shape[0]#gt.shape[0]
                bias = repeat(bias, "1 L -> b L", b=bsz)
            if self.bias_weight==None:
                return dict(logits=vid_feat_stage.squeeze(-1), gt=gt)
            else:
                return dict(logits=vid_feat_stage.squeeze(-1) + bias, gt=gt)

