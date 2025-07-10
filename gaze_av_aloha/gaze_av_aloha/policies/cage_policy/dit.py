import torch
from torch import nn, Tensor
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
import math
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class SinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, dim, num_heads, dropout):
        super().__init__()

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            query_dim=dim, 
            heads=num_heads, 
            dim_head=dim//num_heads, 
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.xattn = Attention(
            query_dim=dim, 
            cross_attention_dim=dim, 
            heads=num_heads, 
            dim_head=dim//num_heads, 
            dropout=dropout
        )
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, dropout=dropout)

    def forward(self, x: Tensor, c: Tensor, t: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.xattn(self.norm2(x), c)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
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
    def __init__(
        self,
        dim=512,
        out_dim=21,
        depth=8,
        num_heads=8,
        dropout=0.1,
        time_dim=256,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(dim, out_dim)
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

    def forward(self, sample: Tensor, condition: Tensor, timestep: Tensor) -> Tensor:
        """
        Args:
            sample: (B, L, dim) - input noise tensor
            condition: (B, L, dim) - conditioning tensor
            timestep: (B,) - tensor of timesteps
        Returns:
            x: (B, L, out_dim) - output tensor after processing through the DiT blocks and final layer
        """
        x = sample
        c = condition
        t = self.time_embed(timestep)
        for block in self.blocks:
            x = block(x, c, t)
        x = self.final_layer(x, t)  
        return x