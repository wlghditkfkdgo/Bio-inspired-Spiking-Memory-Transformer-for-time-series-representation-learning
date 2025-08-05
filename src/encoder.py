import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, surrogate

from positional import tAPE
from layers import SSA_rel_scl, MLP, MutualCrossAttention, SpikLinearLayer

__all__ = ['spikformer']

# class Split2Patch(nn.Module):
#     def __init__(self, patch_size=3, stride=3, padding_patches=None) -> None:
#         """
#         input: [T B L]
#         output: [T B N P]    
#         """
#         super().__init__()
        
#         self.patch_size = patch_size
#         self.stride = stride
        
#         self.padding_patches = padding_patches
        
#         if padding_patches == "end":
#             self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

#     def forward(self, z):
#         # T, B, L = z.shape
        
#         if self.padding_patches == "end":
#             z = self.padding_patch_layer(z)
        
#         z = z.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        
#         return z


class Embedding(nn.Module):
    def __init__(self, num_patches, pe=False, patch_size=63, stride=2, embed_dim=128, in_channel=3, dropout=0, bias=False, tau=2.0) -> None:
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        # in_channel = patch_size
        
        self.stride = stride
        self.embed_dim = embed_dim
        self.pe = pe
        
        self.emb_conv = nn.Conv1d(in_channel, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)
        self.emb_bn = nn.BatchNorm1d(embed_dim)
        self.emb_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # positional encoding
        if pe:
            self.tape = tAPE(embed_dim, max_len=num_patches)
            self.ape_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # Residual dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.tensor, pe=True): 
        T, B, N, C = x.shape
        
        x = self.emb_conv(x.flatten(0, 1)) # have some fire value [TB C N1 N2]
        # x = self.emb1_linear(x.flatten(0, 1))
        x = self.emb_bn(x)
        x = x.reshape(T, B, self.embed_dim, -1).contiguous()
        self.org_x = x.clone().detach().transpose(-1, -2) # [T B N D]
        x = x.flatten(3).contiguous() # [T B D N] 

        if pe:
            x = self.tape(x)
            
        
        x = self.emb_lif(x.reshape(T, B, self.embed_dim, -1).contiguous())
        x = x.transpose(-1, -2).contiguous()  # [T B N D]
        
        return x

    
class Block(nn.Module):
    def __init__(self, dim, seq_len, num_heads, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0.1, norm_layer=nn.LayerNorm, sr_ratio=1, lif_bias=False, tau=2.0, mutual=False, attn='MSSA'):
        super().__init__()
        
        self.attn = SSA_rel_scl(dim, seq_len, num_heads, lif_bias=lif_bias, tau=tau, attn=attn, drop=drop) 
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mutual: self.mca = MutualCrossAttention(dim=dim, lif_bias=lif_bias, num_heads=num_heads, tau=tau, attn=attn, drop=drop)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, lif_bias=lif_bias, tau=tau)
        
        self.gate = nn.Linear(seq_len * 2, seq_len, bias=lif_bias)
        self.gate_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend='cupy')
        
        # self.dropout = nn.Dropout(drop_path)

        
    def forward(self, x: torch.Tensor, mx=None, time_block:nn.Module=None):
        T, B, N, D = x.shape

        x = x * (1. - self.attn(x))
        if mx is not None: x = x * (1. - self.mca(x, mx)) 
        x = x * (1. - self.mlp(x)) # channel mixer
        
        return x
  
class TemporalBlock(nn.Module):
    def __init__(self, num_layers, T, patch_size=[16, 32], embed_dim=[64, 128, 256], ratio=8, lif_bias=False, tau=2.0, num_heads=8, mutual=False):
        super().__init__()
        
        self.T = T  # time step

        self.patch_size = patch_size
        self.num_layers = num_layers
        
        self.in_layer = SpikLinearLayer(embed_dim, embed_dim * 2, lif_bias=lif_bias, tau=tau)
        self.layers = nn.ModuleList([SpikLinearLayer(embed_dim * 2, embed_dim * 2, lif_bias=lif_bias, tau=tau) for l in range(num_layers)])
        self.out_layer = SpikLinearLayer(embed_dim * 2, embed_dim, lif_bias=lif_bias, tau=tau)

    def forward(self, x):
        

        x = self.in_layer(x)
        for l in range(self.num_layers):
            x = x * (1. - self.layers[l](x))
        x = self.out_layer(x)

        return x
