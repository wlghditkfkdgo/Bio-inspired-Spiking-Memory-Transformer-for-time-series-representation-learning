import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg 
from timm.models.layers import trunc_normal_
# from timm.optim.optim_factory import create_optimizer_v2

# from timm.models import create_model
from timm.utils import *

from layers import SpkEncoder
from encoder import Embedding, Block, TemporalBlock
from layers import SpkEncoder, MemoryUpdate, SpikLinearLayer
from utils import random_masking_3D

import math


__all__ = ['spikformer']

class Decoder(nn.Module):
    def __init__(self, embed_dim, patch_size=None, d_ff=2, tau=2.0, bias=True) -> None:
        super().__init__()
        
        self.replay = MemoryUpdate(dim=embed_dim, out_seq=True, lif_bias=bias, tau=tau)

        d_ff = embed_dim
        
        patch_size = patch_size or embed_dim
        
        self.recon2 = nn.Linear(d_ff, patch_size, bias=bias)
        
    def forward(self, x, tr_mx=None):
        """
        [original] x: N x L x C(embed_dim)
        [MyModel] x: T x B x D
        
        out: reconstructed output -> N x L x c_out
        if expand is True: out's shape becomes [B X L]
        """
        
        T, B, N, D = x.shape

        tr_mx = self.replay(x, tr_mx)
        x = x * (1. - tr_mx)
        x = self.recon2(x.flatten(0, 1))
        rec_x = x.reshape(T, B, N, -1).contiguous()
        
        return rec_x

class Spikformer(nn.Module):
    def __init__(self, gating=['original', 'ablation'], 
                 train_mode=['pretraining', 'training', 'testing', 'visual'], 
                 window_size=192,
                data_patch_size=63, 
                num_classes=2, 
                time_num_layers=2,
                embed_dim=[64, 128, 256], 
                num_heads=[1, 2, 4], 
                mlp_ratios=1, 
                qkv_bias=False, 
                qk_scale=None,
                drop_rate=0., 
                attn_drop_rate=0., 
                drop_path_rate=0., 
                norm_layer=nn.LayerNorm,
                depths=[6, 8, 6], 
                sr_ratios=[8, 4, 2], 
                T = 4, 
                num_channels=3,
                data_patching_stride=2, 
                padding_patches=None, 
                lif_bias=False, 
                tau=2.0, 
                spk_encoding=False,
                attn=['SSA', 'MSSA'],
                pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None,
                ):
        super().__init__()
        
        self.train_mode = train_mode
        self.gating = gating
        
        self.T = T  # time step
        self.spk_encoding = spk_encoding
        self.data_patch_size = data_patch_size
        self.data_patching_stride = data_patch_size // 2
        self.num_channels = num_channels
        
        mlp_ratios = 2
        
        self.num_patches = int((window_size - data_patch_size) / (self.data_patching_stride) + 1)
        self.T = self.num_patches
        
        print(f"len of tokens in sequence >> {self.num_patches}")


        self.spk_encoder = SpkEncoder(T) if spk_encoding else None
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        self.encoding = Embedding(num_patches=self.num_patches,
                                    patch_size=self.data_patch_size,
                                    embed_dim=embed_dim,
                                    stride=self.data_patching_stride,
                                    in_channel=num_channels,
                                    pe=True,
                                    bias=lif_bias,
                                    tau=tau
                                    )

        self.data_block = nn.ModuleList([Block(
                    dim=embed_dim, seq_len=self.num_patches, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j], attn=attn,
                    norm_layer=norm_layer, sr_ratio=sr_ratios, lif_bias=lif_bias, tau=tau, mutual=True)
                    for j in range(depths)])
        
        self.time_block = TemporalBlock(T=T,
                                        num_layers=time_num_layers,
                                        patch_size=data_patch_size,
                                        embed_dim=embed_dim,
                                        lif_bias=lif_bias,
                                        num_heads=num_heads,
                                        tau=tau,
                                        mutual=False)
        
        if ((self.train_mode == 'training') or (self.train_mode == 'pre_training')) : 
            self.weak_decoder = Decoder(embed_dim=embed_dim, patch_size=None, tau=tau, bias=lif_bias)
            
        self.head = nn.Linear(embed_dim, num_classes, bias=lif_bias) if num_classes > 0 else nn.Identity()

        if gating == 'ablation':
            self._init_ablation()

        self.apply(self._init_weights)
        
        if ((self.train_mode != 'training') and (self.train_mode != 'pre_training')):
            for module in self.modules():
                if isinstance(module, nn.BatchNorm1d):
                    module.eval()

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, N):
        if N == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(N), mode="bilinear").reshape(1, -1, N).permute(0, 2, 1)

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _gen_mask(self, x) -> tuple:
        """Generating boolean mask which have same dimension with `x`

        Args:
            x (torch.tensor): embedding matrix after PE ([T B N D])
            

        - Return:
            x_data (torch.tensor): input for hippo.
            temporal_emb (torch.tensor): input for neocortix.
            
        """
        T, B, N, D = x.shape
        
        x_masked, _, self.mask, _ = random_masking_3D(x.flatten(0, 1), mask_ratio=0.6)
        self.mask = self.mask.float().reshape(T, B, -1).contiguous().unsqueeze(-1) #[TB N]
        x_masked = x_masked.reshape(T, B, N, -1).contiguous()
        
        x_t = x.transpose(0, 2).contiguous()
        x_t_masked, _, self.time_mask, _ = random_masking_3D(x_t.flatten(0, 1), mask_ratio=0.6)
        self.time_mask = self.time_mask.float().reshape(T, B, -1).contiguous().unsqueeze(-1) #[NB T]
        x_t_masked = x_t_masked.reshape(T, B, N, -1).contiguous()
        
        x_data, temporal_emb = x_masked, x_t_masked.mean(2, keepdim=True)
        
        return x_data, temporal_emb
        
    def forward(self, x):
        if self.spk_encoding:
            x = self.spk_encoder(x)
        else: 
            x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1) 

        x = self.encoding(x) # [T B N D]
        self.org_x = self.encoding.org_x.clone().detach().transpose(0, 2).contiguous().mean(2, keepdim=True)
        # [N B T D]
        
        if self.train_mode == 'pre_training' :
            x_data, temporal_emb = self._gen_mask(x)
        else:
            temporal_emb = x.transpose(0, 2).contiguous().mean(2, keepdim=True)
            x_data = x
        
        temporal_emb = self.time_block(temporal_emb) if (self.gating != 'ablation') else None
        
        for dblk in self.data_block:
            x_data = dblk(x_data, temporal_emb)
        
        if (self.gating != 'ablation') and ((self.train_mode == 'training') or (self.train_mode == 'pre_training')):
            
            return self._training(x_data, temporal_emb) 
            
        elif (self.train_mode == 'testing') or (self.gating == 'ablation'):
            # self._init_ablation()
            if self.train_mode == 'pre_training' :
                loss = ((x_data - self.org_x.transpose(0, 2).contiguous()) ** 2)
                mse = (loss * self.mask).sum() / self.mask.sum()
                return mse
            else:
                return self._testing(x_data)
        
        elif self.train_mode == 'visual':

            return self._visualization(x_data, temporal_emb)
        
        else:
            raise ValueError
        
        
    def _training(self, x_data, temporal_emb):
        rec_x = self.weak_decoder(temporal_emb, x_data) #[T B 1 P]
        
        self.rec_x_fr = rec_x.mean(0)
        
        assert self.org_x.shape == rec_x.shape, f"original x_time' shape is {self.org_x.shape}, and reconstructed x_time' shape is {rec_x.shape}"

        if self.train_mode == 'training' : 
            z = self.head(x_data.mean(2)).mean(0)
            return z, x_data.clone().detach().transpose(0, 2).contiguous(), temporal_emb, self.org_x, rec_x
        
        elif self.train_mode == 'pre_training' :
            loss1 = ((rec_x - self.org_x) ** 2)
            mse = (loss1 * self.mask.transpose(0, 2).contiguous() * self.time_mask).sum() / (self.mask.sum() * self.time_mask.sum())
            return mse
    
    def _testing(self, x_data):
        
        return self.head(x_data.mean(0).mean(1))
    

    # TODO
    def _init_ablation(self):
        
        delattr(self, 'time_block')
        # delattr(self, 'gate_attn')
        if hasattr(self, 'weak_decoder'): delattr(self, 'weak_decoder')
        if hasattr(self, 'replay') : delattr(self, 'replay')
    

@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(pretrained=pretrained, **kwargs)
    model.default_cfg = _cfg()
    return model