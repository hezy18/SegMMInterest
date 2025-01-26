import torch
import torch.nn as nn
import torch.nn.functional as F
from kn_util.nn_utils import clones
from einops import einsum, rearrange, repeat, reduce
import numpy as np
from kn_util.nn_utils.layers import MLP
#from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from torch.nn import LayerNorm
import time

class SegFormerXAttention(nn.Module):

    def __init__(self, d_model, num_head, sr_ratio=1, dropout=0.1, ablation_type='ours') -> None:
        super().__init__()
        d_head = d_model // num_head
        self.t2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2v_proj = clones(nn.Linear(d_model, d_model), 3)
        self.t2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.v2t_proj = clones(nn.Linear(d_model, d_model), 3)
        self.do = nn.Dropout(dropout)

        if sr_ratio > 1.0:
            self.sr = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=sr_ratio,
                stride=sr_ratio,
                padding=(sr_ratio - 1) // 2,
            )
        self.sr_ratio = sr_ratio

        self.d_head = d_head
        self.num_head = num_head

        self.ff_usr = nn.Linear(d_model, d_model)
        self.ff_vid = nn.Linear(d_model, d_model)

        self.ln_usr = LayerNorm(d_model, 1e-12)
        self.ln_vid = LayerNorm(d_model, 1e-12)

        self.ablation_type = ablation_type

    def get_attn_logits(self, feat_k, mask_k, feat_q, mask_q, proj): 
        B, Lq, _ = feat_q.shape
        B, Lk, _ = feat_k.shape
        B, Lk = mask_k.shape
        B, Lq = mask_q.shape
        
        feat_q = rearrange(
            proj[0](feat_q), # torch.Size([6, 20, 512])
            "b lq (h dh) -> b lq h dh",
            h=self.num_head,
            dh=self.d_head,
        ) # torch.Size([6, 20, 16, 32])
        
        feat_k = rearrange(
            proj[1](feat_k),
            "b lk (h dh) -> b lk h dh",
            h=self.num_head,
            dh=self.d_head,
        )
        # print('print shape', feat_q.shape, feat_k.shape)
        attn_logits = einsum(feat_q, feat_k, "b lq h dh, b lk h dh->b h lq lk")
        attn_mask = repeat(
            einsum(mask_q, mask_k, "b lq, b lk->b lq lk"),
            "b lq lk->b h lq lk",
            h=self.num_head,
        )

        attn_logits[~attn_mask] = -10000.0

        return attn_logits  # b lv h lq lk

    def forward(self, vid_feat, vid_mask, usr_feat, usr_mask):
        # B, Lv, _ = vid_feat.shape
        # B, Lt, _ = usr_feat.shape
        B, Lv = vid_mask.shape
        B, Lt = usr_mask.shape

        #vid_feat_ = vid_feat # incorrect implementation of residual connection
        #txt_feat_ = txt_feat

        if self.sr_ratio > 1.0:
            vid_feat_sr = vid_feat.transpose(1, 2)
            vid_feat_sr = self.sr(vid_feat_sr)
            vid_feat_sr = vid_feat_sr.transpose(1, 2)
        else:
            vid_feat_sr = vid_feat

        vid_mask_sr = vid_mask[:, None, :]
        vid_mask_sr = nn.MaxPool1d(kernel_size=self.sr_ratio, stride=self.sr_ratio)(vid_mask_sr.float())
        vid_mask_sr = (vid_mask_sr > 0)[:, 0, :]

        v2v_value = self.v2v_proj[2](vid_feat_sr)
        t2v_value = self.t2v_proj[2](usr_feat)
        v2t_value = self.v2t_proj[2](vid_feat_sr)
        t2t_value = self.t2t_proj[2](usr_feat)

        v2v_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, vid_feat, vid_mask, self.v2v_proj)
        t2v_logits = self.get_attn_logits(usr_feat, usr_mask, vid_feat, vid_mask, self.t2v_proj)
        v2t_logits = self.get_attn_logits(vid_feat_sr, vid_mask_sr, usr_feat, usr_mask, self.v2t_proj)
        try:
            t2t_logits = self.get_attn_logits(usr_feat, usr_mask, usr_feat, usr_mask, self.t2t_proj)
        except:
            import pdb; pdb.set_trace()

        if 'CrossAtt' in self.ablation_type:
            v_value = t2v_value
            v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

            t_value = v2t_value
            t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

            v_logits = t2v_logits
            v_logits = self.do(v_logits)
            v_logits /= np.sqrt(self.d_head)
            
            t_logits = v2t_logits
            t_logits = self.do(t_logits)
            t_logits /= np.sqrt(self.d_head)
        elif 'SelfAtt' in self.ablation_type:
            v_value = v2v_value
            v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

            t_value = t2t_value
            t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

            v_logits = v2v_logits
            v_logits = self.do(v_logits)
            v_logits /= np.sqrt(self.d_head)
            
            t_logits = t2t_logits
            t_logits = self.do(t_logits)
            t_logits /= np.sqrt(self.d_head)

        else:
            v_value = torch.cat([v2v_value, t2v_value], dim=1)
            v_value = rearrange(v_value, "b lk (h dh)->b lk h dh", h=self.num_head)

            t_value = torch.cat([v2t_value, t2t_value], dim=1)
            t_value = rearrange(t_value, "b lk (h dh)->b lk h dh", h=self.num_head)

            v_logits = torch.cat([v2v_logits, t2v_logits], dim=-1)
            v_logits = self.do(v_logits)
            v_logits /= np.sqrt(self.d_head)
            
            t_logits = torch.cat([v2t_logits, t2t_logits], dim=-1)
            t_logits = self.do(t_logits)
            t_logits /= np.sqrt(self.d_head)

        vid_feat_ = einsum(
            F.softmax(v_logits, dim=-1),
            v_value,
            "b h lq lk, b lk h d -> b lq h d",
        )
        usr_feat_ = einsum(
            F.softmax(t_logits, dim=-1),
            t_value,
            "b h lq lk, b lk h d -> b lq h d",
        )

        vid_feat_ = rearrange(vid_feat_, "b lq h d -> b lq (h d)")
        usr_feat_ = rearrange(usr_feat_, "b lq h d -> b lq (h d)")

        usr_feat_ = self.do(self.ff_usr(usr_feat_))
        vid_feat_ = self.do(self.ff_vid(vid_feat_))


        usr_feat = self.ln_usr(usr_feat + usr_feat_)
        vid_feat = self.ln_vid(vid_feat + vid_feat_)
        if 'SelfAtt' in self.ablation_type:
            return vid_feat, None
        else:
            return vid_feat, usr_feat


class SegFormerXEncoderLayer(nn.Module):

    def __init__(self, d_model, num_head, ff_dim, sr_ratio, dropout, ablation_type='ours') -> None:
        super().__init__()
        self.cross_attn = SegFormerXAttention(d_model=d_model, num_head=num_head, sr_ratio=sr_ratio, dropout=dropout, ablation_type=ablation_type)
        self.ff_usr = MLP([d_model, ff_dim, d_model], activation="gelu")
        self.ff_vid = MLP([d_model, ff_dim, d_model], activation="gelu")
        self.ln_usr = LayerNorm(d_model, eps=1e-12)
        self.ln_vid = LayerNorm(d_model, eps=1e-12)
        self.do = nn.Dropout(dropout)

    def forward(self, usr_feat, usr_mask, vid_feat, vid_mask):
        """
        x = temporal_attn(x) + x
        x = cross_attn(x) + x
        x = OUTPUT(x)
        """
        B, Lv, _ = vid_feat.shape

        vid_feat, usr_feat = self.cross_attn(vid_feat, vid_mask, usr_feat, usr_mask)

        #vid_feat_ = vid_feat # ? incorrect implementation of residual connection
        #usr_feat_ = usr_feat

        vid_feat_ = self.do(self.ff_vid(vid_feat))
        vid_feat = self.ln_vid(vid_feat + vid_feat_)
        if usr_feat!=None:
            usr_feat_ = self.do(self.ff_usr(usr_feat))
            usr_feat = self.ln_usr(usr_feat + usr_feat_)

        return vid_feat, usr_feat

class MLP_Block(nn.Module):
	'''
	Reference: FuxiCTR
	https://github.com/reczoo/FuxiCTR/blob/v2.0.1/fuxictr/pytorch/layers/blocks/mlp_block.py
	'''
	def __init__(self, input_dim, hidden_units=[], 
				 hidden_activations="ReLU", output_dim=None, output_activation=None, 
				 dropout_rates=0.0, batch_norm=False, 
				 layer_norm=False, norm_before_activation=True,
				 use_bias=True):
		super(MLP_Block, self).__init__()
		dense_layers = []
		if not isinstance(dropout_rates, list):
			dropout_rates = [dropout_rates] * len(hidden_units)
		if not isinstance(hidden_activations, list):
			hidden_activations = [hidden_activations] * len(hidden_units)
		hidden_activations = [getattr(nn, activation)()
                    if activation != "Dice" else Dice(emb_size) for activation, emb_size in zip(hidden_activations, hidden_units)]
		hidden_units = [input_dim] + hidden_units
		for idx in range(len(hidden_units) - 1):
			dense_layers.append(nn.Linear(hidden_units[idx], hidden_units[idx + 1], bias=use_bias))
			if norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if hidden_activations[idx]:
				dense_layers.append(hidden_activations[idx])
			if not norm_before_activation:
				if batch_norm:
					dense_layers.append(nn.BatchNorm1d(hidden_units[idx + 1]))
				elif layer_norm:
					dense_layers.append(nn.LayerNorm(hidden_units[idx + 1]))
			if dropout_rates[idx] > 0:
				dense_layers.append(nn.Dropout(p=dropout_rates[idx]))
		if output_dim is not None:
			dense_layers.append(nn.Linear(hidden_units[-1], output_dim, bias=use_bias))
		if output_activation is not None:
			dense_layers.append(getattr(nn, output_activation)())
		self.mlp = nn.Sequential(*dense_layers) # * used to unpack list
	
	def forward(self, inputs):
		return self.mlp(inputs)

class SegFormerXEncoder(nn.Module): # 1

    def __init__(self, d_model_in, d_model_lvls, num_head_lvls, sr_ratio_lvls, ff_dim_lvls, use_patch_merge,
                 dropout, ablation_type='ours') -> None:
        super().__init__()
        assert (len(d_model_lvls) == len(num_head_lvls) == len(sr_ratio_lvls) == len(ff_dim_lvls))
        self.layers = nn.ModuleList([
                SegFormerXEncoderLayer(d_model=d_model,
                                    num_head=num_head,
                                    ff_dim=ff_dim,
                                    sr_ratio=sr_ratio,
                                    dropout=dropout,
                                    ablation_type=ablation_type)
                for d_model, num_head, sr_ratio, ff_dim in zip(d_model_lvls, num_head_lvls, sr_ratio_lvls, ff_dim_lvls)
            ])

        d_model_lvls_ = [d_model_in] + d_model_lvls
        self.pe_lns = nn.ModuleList([nn.LayerNorm(d_model_lvls[i], 1e-12) for i in range(len(d_model_lvls))])
        self.txt_lvl_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model_lvls_[i - 1], d_model_lvls_[i]),
                LayerNorm(d_model_lvls_[i], eps=1e-12),
            ) for i in range(1, len(d_model_lvls_))
        ])
        self.use_patch_merge = use_patch_merge
        self.patch_merge = nn.ModuleList([
            nn.Conv1d(in_channels=d_model_lvls_[i - 1],
                      out_channels=d_model_lvls_[i],
                      kernel_size=3,
                      stride=2,
                      padding=1) for i in range(1, len(d_model_lvls_))
        ])

    @staticmethod
    def _interpolate_to_same_size(x, size):
        # B, L, D = x.shape
        x = x.transpose(1, 2)
        # x = F.interpolate(x, size=size, mode="linear")
        x = F.avg_pool1d(x, kernel_size=2, stride=2)
        x = x.transpose(1, 2)
        return x

    def _patch_merge(self, vid_feat, idx):
        vid_feat = vid_feat.transpose(1, 2)
        vid_feat = self.patch_merge[idx](vid_feat)
        vid_feat = vid_feat.transpose(1, 2)
        return vid_feat

    def forward(self, usr_feat, usr_mask, vid_feat, vid_mask, vid_pe=None): 
        # vid_pe not None -> add vid_pe to each level of vid_feat
        B, Lv, _ = vid_feat.shape
        intermediate_states = []
        intermediate_masks = []
        for idx, layer in enumerate(self.layers):
            if self.use_patch_merge[idx]: 
                vid_feat = self._patch_merge(vid_feat, idx=idx)
                vid_mask = self._interpolate_to_same_size(vid_mask[:, :, None].float(),
                                                          vid_feat.shape[1]).squeeze(-1).bool()
                if vid_pe is not None:
                    vid_pe = self._interpolate_to_same_size(vid_pe, vid_feat.shape[1])
                    vid_feat = self.pe_lns[idx](vid_pe + vid_feat)

            intermediate_states += [vid_feat]
            intermediate_masks += [vid_mask]
            #txt_feat = self.txt_lvl_projs[idx](txt_feat) # ?
            vid_feat, usr_feat_test = layer(usr_feat, usr_mask, vid_feat, vid_mask)
            if usr_feat_test!=None:
                usr_feat = usr_feat_test
        # import pdb; pdb.set_trace()
        
        return intermediate_states, intermediate_masks


class SegFormerX(nn.Module):
    """Video-text version of Segformer"""

    def __init__(self,
                 d_model_in=128,
                 d_model_lvls=[128, 256, 512, 1024],
                 num_head_lvls=[2, 4, 8, 16],
                 ff_dim_lvls=[256, 512, 1024, 2048],
                 sr_ratio_lvls=[8, 4, 2, 1],
                 input_vid_dim=768,
                 input_usr_dim=768,
                 #input_txt_dim=768,
                 max_vid_len=256,
                 #max_txt_len=30,
                 max_usr_len=20,
                 dropout=0.1,
                 # pe="learn",
                 pe_kernel_size=3,
                 use_patch_merge=[True, False, True, False],
                 output_layers=None,
                 model_cfg=None,
                 user_id_max = -1,
                 video_id_max = -1,
                 use_pe = 1) -> None:
        super().__init__()
        if video_id_max!=-1:
            self.vid_proj = nn.Embedding(video_id_max+1, d_model_in//2)
            self.frameid_proj = nn.Linear(1, d_model_in//2)
        else:
            self.vid_proj = nn.Linear(input_vid_dim, d_model_in)
        
        #self.txt_proj = nn.Linear(input_txt_dim, d_model_in)
        if user_id_max!=-1:
            self.usr_proj = nn.Embedding(user_id_max+1, d_model_in)
        else:
            self.usr_proj = nn.Linear(input_usr_dim, d_model_in)
        #self.mask_embedding = nn.Embedding(1, d_model_in)
        self.debug = model_cfg.debug
        self.num_layers_enc = model_cfg.num_layers_enc
        # self.pe = pe
        self.use_pe = use_pe
        # if pe == "learn":
        self.vid_pe = nn.Embedding(max_vid_len, d_model_in)
        # elif pe == "conv":
        #     self.conv_pe = nn.Conv1d(in_channels=d_model_in,
        #                              out_channels=d_model_in,
        #                              kernel_size=pe_kernel_size,
        #                              padding=(pe_kernel_size - 1) // 2)
        # else:
        #     raise NotImplementedError()
        #self.txt_pe = nn.Embedding(max_txt_len, d_model_in)
        self.usr_pe = nn.Embedding(max_usr_len, d_model_in)
        # self.type_embed_vid = nn.Parameter(torch.randn(d_model_in))
        # self.type_embed_vid.data.normal_(mean=0.0, std=0.01)
        # self.type_embed_txt = nn.Parameter(torch.randn(d_model_in))
        # self.type_embed_txt.data.normal_(mean=0.0, std=0.02)
        self.vid_ln = LayerNorm(d_model_in, eps=1e-12)
        #self.txt_ln = LayerNorm(d_model_in, eps=1e-12)
        self.usr_ln = LayerNorm(d_model_in, eps=1e-12)
        self.do = nn.Dropout(dropout)
        if hasattr(model_cfg, 'ablation_type'):
            self.ablation_type = model_cfg.ablation_type
        else:
            self.ablation_type = 'ours'

        if self.ablation_type=='CrossMLP':
            self.encoder_mlp = MLP_Block(input_dim=d_model_lvls[0], output_dim=d_model_lvls[0], 
                        hidden_units=d_model_lvls[2:-2], hidden_activations='ReLU', 
                        dropout_rates=dropout, batch_norm=0)
            self.encoder_pooling = nn.AdaptiveAvgPool1d(40)
        elif self.ablation_type=='SelfMLP' or self.ablation_type=='w/oAtt':
            self.encoder_mlp = MLP_Block(input_dim=d_model_lvls[0], output_dim=d_model_lvls[0], 
                        hidden_units=d_model_lvls[1:-1], hidden_activations='ReLU', 
                        dropout_rates=dropout, batch_norm=0)
        else:
            self.encoder = SegFormerXEncoder(d_model_in=d_model_in,
                                            d_model_lvls=d_model_lvls,
                                            num_head_lvls=num_head_lvls,
                                            sr_ratio_lvls=sr_ratio_lvls,
                                            ff_dim_lvls=ff_dim_lvls,
                                            dropout=dropout,
                                            use_patch_merge=use_patch_merge,
                                            ablation_type=self.ablation_type)
        self.output_layers = [_ for _ in range(len(sr_ratio_lvls))] if output_layers is None else output_layers

        self.apply(self.init_weight)

    def init_weight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_embedding(self, usr_feat, vid_feat):
        if vid_feat.ndim==2:
            B, Lv = vid_feat.shape
            if 'noPos' in self.ablation_type:
                frame_positions = torch.stack([torch.randperm(Lv) for _ in range(B)]).float().to(vid_feat.device)
            else:
                frame_positions = torch.arange(Lv).unsqueeze(0).repeat(B, 1).float().to(vid_feat.device)  # batch_size * clip_num
            vid_feat = self.vid_proj(vid_feat) # B,40,64
            
            frame_feat= self.frameid_proj(frame_positions.unsqueeze(-1)) # B,40,64
            vid_feat = torch.cat([vid_feat, frame_feat], dim=-1) # B,40,128
        else:
            B, Lv, _ = vid_feat.shape
            vid_feat = self.vid_proj(vid_feat)
            

        if usr_feat.ndim==2:
            B, Lt = usr_feat.shape
        else:
            B, Lt, _ = usr_feat.shape
        usr_feat = self.usr_proj(usr_feat) # B,1,128

        # import pdb; pdb.set_trace()

        # if self.pe == "learn":
        if self.use_pe:
            vid_pe = self.vid_pe.weight[None, :vid_feat.shape[1]] # about position embedding
            # elif self.pe == "conv":
            #     pe = self.conv_pe(vid_feat.transpose(1, 2))
            #     pe = pe.transpose(1, 2)
            vid_feat = self.vid_ln(vid_feat + vid_pe) # about position embedding(不加就可以了吧) 
            # vid_feat = vid_feat + vid_pe
        else:
            vid_pe=None
            vid_feat = self.vid_ln(vid_feat)
            # vid_feat = vid_feat
        vid_feat = self.do(vid_feat)
        
        if self.use_pe:
            usr_pe = self.usr_pe.weight[None, :usr_feat.shape[1]]
            usr_feat = self.usr_ln(usr_feat + usr_pe) 
            # import pdb; pdb.set_trace()
            # usr_feat = usr_feat + usr_pe
        else:
            usr_feat = self.usr_ln(usr_feat) 
            # usr_feat = usr_feat
        usr_feat = self.do(usr_feat)

        return vid_feat, usr_feat, vid_pe  # 去掉pe 则为None

    def forward(self, usr_feat, usr_mask, vid_feat, vid_mask):
        # print(type(usr_feat), usr_feat.shape)
        # print(type(vid_feat), vid_feat.shape)
        if usr_feat.ndim==1:
            usr_feat = usr_feat.unsqueeze(-1)
            B, Lt = usr_feat.shape
            usr_mask = torch.ones((B, Lt)).to(usr_feat.device)
        else:
            B, Lt, _ = usr_feat.shape
        if vid_feat.ndim==1:
            Lv = 40
            vid_feat = vid_feat.unsqueeze(-1).repeat(1,Lv)
        else:
            B, Lv, _ = vid_feat.shape
        usr_mask = usr_mask.bool()
        if self.debug:
            st = time.time()
        
        # import pdb;pdb.set_trace();
        vid_feat, usr_feat, vid_pe = self._get_embedding(usr_feat, vid_feat)
        #TODO:change: use parameter
        # vid_mask = torch.ones(vid_feat.shape[:2], dtype=torch.bool, device=usr_feat.device)
        
        if self.debug:
            et = time.time()
            print(f"get embedding time cost: {et-st}")
            st = time.time()

        if self.ablation_type=='CrossMLP':
            cross_mlp_out = self.encoder_mlp(torch.cat((usr_feat, vid_feat), dim=-2))
            vid_results = self.encoder_pooling(cross_mlp_out.permute(0, 2, 1))
            selected_interm_state = [vid_results.permute(0, 2, 1)]
        elif self.ablation_type=='SelfMLP':
            self_mlp_out = self.encoder_mlp(vid_feat)
            selected_interm_state = [self_mlp_out]
        elif self.ablation_type=='w/oAtt':
            selected_interm_state = [vid_feat]
        else:
            intermediate_states, intermediate_masks = self.encoder(usr_feat, usr_mask, vid_feat, vid_mask, vid_pe) # 这里用到了
            selected_interm_state = [intermediate_states[i] for i in self.output_layers]

        if self.debug:
            et = time.time()
            print(f"get transformer cost: {et-st} with {(et-st)/self.num_layers_enc} per layer")

        return selected_interm_state, usr_feat


class SegFormerXFPN(nn.Module):

    def __init__(self,
                 backbone,
                 output_layer=[0, 2, 3],
                 intermediate_hidden_size=[512, 512, 512],
                 fpn_hidden_size=512) -> None:
        super().__init__()
        self.backbone = backbone
        self.output_layer = output_layer
        self.adapters = nn.ModuleList([
            nn.Conv1d(in_channels=cur_size, out_channels=fpn_hidden_size, kernel_size=1)
            for cur_size in intermediate_hidden_size
        ])  # adapt to a fixed fpn_hidden_size
        self.out_convs = nn.ModuleList([
            nn.Conv1d(in_channels=fpn_hidden_size, out_channels=fpn_hidden_size, kernel_size=3, padding=1)
            for _ in intermediate_hidden_size
        ])

    def forward(self, vid_feat, txt_feat, txt_mask, word_mask):
        intermediate_states, txt_feat = self.backbone(vid_feat=vid_feat,
                                                      txt_feat=txt_feat,
                                                      txt_mask=txt_mask,
                                                      word_mask=word_mask)
        fpn_states = []

        for idx, adapter in enumerate(self.adapters):
            state_idx = self.output_layer[idx]
            cur_state = intermediate_states[state_idx].transpose(1, 2)
            fpn_states += [adapter(cur_state)]

        for idx in range(0, len(self.adapters) - 1):
            fpn_states[idx] += F.interpolate(fpn_states[idx + 1], fpn_states[idx].shape[-1])

        for idx, fpn_state in enumerate(fpn_states):
            fpn_states[idx] = self.out_convs[idx](fpn_state).transpose(1, 2)

        return fpn_states, txt_feat