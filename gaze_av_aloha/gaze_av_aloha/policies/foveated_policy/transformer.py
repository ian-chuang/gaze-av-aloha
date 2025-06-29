import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from timm.layers import Mlp
import numpy as np
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def maybe_add_pos_embed(tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
    return tensor if pos_embed is None else tensor + pos_embed

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, add_bias_kv=True, batch_first=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.xattn = nn.MultiheadAttention(hidden_size, num_heads, add_bias_kv=True, batch_first=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        q = k = v = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(q, k, v, need_weights=False)[0]

        q = self.norm2(x)
        k = v = c
        x = x + self.xattn(q, k, v, need_weights=False)[0]

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        return x

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        out_dim=21,
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, out_dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c, t):
        for block in self.blocks:
            x = block(x, c, t)
        x = self.final_layer(x, t)  
        return x

class AttentionPoolingBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.xattn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=0)

    def forward(self, x, c, x_pos_emb=None, c_pos_emb=None):
        x_norm = self.norm1(x)
        q = k = maybe_add_pos_embed(x_norm, x_pos_emb)
        v = x_norm
        x = x + self.attn(q, k, v, need_weights=False)[0]

        x_norm = self.norm2(x)
        q = maybe_add_pos_embed(x_norm, x_pos_emb)
        k = maybe_add_pos_embed(c, c_pos_emb)
        v = c
        x = x + self.xattn(q, k, v, need_weights=False)[0]

        x = x + self.mlp(self.norm3(x))

        return x

class AttentionPooling(nn.Module):
    """
    Attention pooling layer that uses DiT as the backbone.
    """
    def __init__(self, 
        num_queries, 
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.query_tokens = nn.Embedding(num_queries, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            AttentionPoolingBlock(
                hidden_size=hidden_size,
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            ) for _ in range(depth)
        ])
        self.norm2 = nn.LayerNorm(hidden_size)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        self.apply(_basic_init)

    def forward(self, c, c_pos_emb=None):
        """
        context: (B, N, D) - Input features, where B is batch size, N is sequence length, and D is feature dimension.
        """
        B = c.size(0)
        x = self.query_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        c = self.norm1(c)
        for block in self.blocks:
            x = block(x=x, c=c, c_pos_emb=c_pos_emb)
        x = self.norm2(x)
        return x 
    

