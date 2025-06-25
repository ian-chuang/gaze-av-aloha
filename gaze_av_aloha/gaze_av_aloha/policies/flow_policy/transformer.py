import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from timm.layers import Mlp

class RotaryPosEmbed(nn.Module):
    """ RoPE implementation from torchtune """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._rope_init()

    def _rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self._build_rope_cache(self.max_seq_len)

    def _build_rope_cache(self, max_seq_len: int = 256) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)

        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x) -> torch.Tensor:
        """
        Inputs: x: [B, num_heads, S, head_dim]
        Returns: [B, num_heads, S, head_dim]
        """
        x = x.permute(0, 2, 1, 3)  # [B, S, num_heads, head_dim]
        B, S, num_heads, head_dim = x.size()

        rope_cache = (self.cache[:S])
        xshaped = x.float().reshape(*x.shape[:-1], head_dim // 2, 2)
        rope_cache = rope_cache.view(1, S, num_heads, head_dim // 2, 2)

        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        x_out = x_out.flatten(3)
        x_out = x_out.permute(0, 2, 1, 3)
        return x_out.type_as(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=512,
        qk_norm=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.attn_norm = self.head_dim ** -0.5
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPosEmbed(dim, max_seq_len=max_seq_len)

    def forward(self, x, mask=None):
        B, S, C = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # (B, num_heads, S, head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        max_seq_len=512,
        qk_norm=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = RotaryPosEmbed(dim, max_seq_len=max_seq_len)

    def forward(self, x, context, mask=None):
        """
        x: [B, S, D] - Query input
        context: [B, S_ctx, D] - Key/Value input
        mask: Optional attention mask
        """
        B, S, C = x.shape
        _, S_ctx, _ = context.shape

        # Project queries from x
        q = self.q_proj(x).reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Project keys and values from context
        kv = self.kv_proj(context).reshape(B, S_ctx, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # Each [B, num_heads, S_ctx, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        if self.training:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop)
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, S, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.xattn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t+c.mean(dim=1)).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + self.xattn(self.norm2(x), c) 
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

    def forward(self, x: Tensor, t: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(t+c.mean(dim=1)).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x
    
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
    
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_dim=21,
        out_dim=21,
        cond_dim=512,
        hidden_size=512,
        depth=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
        time_dim=128,
    ):
        super().__init__()

        # Encoder for the diffusion timestep.
        self.x_embed = nn.Linear(in_dim, hidden_size)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, hidden_size),
        )
        self.cond_embed = nn.Linear(cond_dim, hidden_size)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size, 
                num_heads, 
                mlp_ratio=mlp_ratio,
                attn_drop=dropout,
                proj_drop=dropout,
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

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, timestep, cond):
        """
        Forward pass of DiT.
        x: (B, N, D) - Input features, where B is batch size, N is sequence length, and D is feature dimension.
        timestep: (B,) - Timestep indices for the diffusion process.
        cond: (B, D) - Conditioning features, where D is the feature dimension.
        """
        x = self.x_embed(x)  
        t = self.time_embed(timestep)
        c = cond
        for block in self.blocks:
            x = block(x, t, c)
        x = self.final_layer(x, t, c)  
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, use_xattn, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.use_xattn = use_xattn
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, **block_kwargs)
        if use_xattn:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.xattn = CrossAttention(hidden_size, num_heads=num_heads, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), drop=0)

    def forward(self, x, c=None):
        """
        x: [B, S, D] - Query input
        context: [B, S_ctx, D] - Key/Value input
        mask: Optional attention mask
        """
        x = x + self.attn(self.norm1(x))
        if self.use_xattn:
            assert c is not None, "Cross-attention requires conditioning context"
            x = x + self.xattn(self.norm2(x), c)
        x = x + self.mlp(self.norm3(x))
        return x

class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple attention blocks.
    """
    def __init__(self, hidden_size, num_layers, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionBlock(
                use_xattn=False,
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                attn_drop=dropout,
                proj_drop=dropout,
            ) for _ in range(num_layers)
        ])
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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class AttentionPooling(nn.Module):
    """
    Attention pooling layer that uses DiT as the backbone.
    """
    def __init__(self, hidden_size, out_dim, num_queries, depth=2, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.intermediate_dim = out_dim // num_queries
        assert out_dim % num_queries == 0, "out_dim must be divisible by num_queries"
        self.query_tokens = nn.Embedding(num_queries, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            AttentionBlock(
                use_xattn=i % 2 == 0,  # Alternate between self-attention and cross-attention
                hidden_size=hidden_size,
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                attn_drop=dropout,
                proj_drop=dropout,
            ) for i in range(depth)
        ])
        # self.proj = nn.Linear(hidden_size, self.intermediate_dim)
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

    def forward(self, c):
        """
        context: (B, N, D) - Input features, where B is batch size, N is sequence length, and D is feature dimension.
        """
        B = c.size(0)
        x = self.query_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        c = self.norm1(c)
        for block in self.blocks:
            x = block(x=x, c=c)
        # x = self.proj(x)  # (B, num_queries, intermediate_dim)
        x = self.norm2(x)
        return x # .flatten(start_dim=1) 