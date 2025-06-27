import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from timm.layers import Mlp

def get_1d_rotary_embed(dim, pos, base=10000):
    """Create 1D rotary positional embeddings from a grid of positions.

    Args:
        dim: the output dimension for each position.
        pos: a tensor of size (seq_len,) of positions to be encoded.

    Returns:
        thetas: a tensor of size (seq_len, dim) of rotary positional embeddings.
    """
    assert dim % 2 == 0
    thetas = 1.0 / (base ** (torch.arange(0, dim, 2, device=pos.device).float() / dim))
    thetas = pos.unsqueeze(-1) * thetas  # (N, D/2)
    thetas = torch.cat((thetas, thetas), dim=-1)  # (N, D)
    return thetas

def apply_rotary_embed(x, thetas):
    """Rotates the input tensors by the positional embeddings.

    Args:
        x: a tensor of shape (..., seq_len, dim).
        thetas: a tensor of shape (..., seq_len, dim) of positional embeddings.

    Returns:
        x: a tensor of shape (..., seq_len, dim) of the rotated input tensors.
    """
    assert x.shape[-2:] == thetas.shape[-2:]
    x1, x2 = x.chunk(2, dim=-1)
    x_rotate_half = torch.cat([-x2, x1], dim=-1)
    return x * thetas.cos() + x_rotate_half * thetas.sin()

def get_foveated_pos_embed(
    gaze: Tensor,
    image_shape: tuple,
    crop_shape: tuple,
    feat_shape: tuple,
    n_obs_steps: int = 1,
    dim: int = 384,
    base: int = 10000,
):
    patch_shape = (crop_shape[0] / feat_shape[0], crop_shape[1] / feat_shape[1])
    num_axis = 3
    assert dim % num_axis == 0
    axis_dim = dim // num_axis
    assert axis_dim % 2 == 0

    # Create meshgrid along eash axis
    axis_grids = list(torch.meshgrid(
        torch.arange(n_obs_steps, dtype=torch.float32, device=gaze.device),
        torch.linspace(-crop_shape[0]/2 + patch_shape[0]/2, crop_shape[0]/2 - patch_shape[0]/2, feat_shape[0], device=gaze.device),
        torch.linspace(-crop_shape[1]/2 + patch_shape[1]/2, crop_shape[1]/2 - patch_shape[1]/2, feat_shape[1], device=gaze.device),
        indexing="ij",
    ))

    gaze_x = gaze[..., 0] * image_shape[1] / 2 # (batch size , n_obs_steps)
    gaze_y = gaze[..., 1] * image_shape[0] / 2 # (batch size , n_obs_steps)

    axis_grids[0] = axis_grids[0].unsqueeze(0).expand(gaze.shape[0], -1, -1, -1)  # Expand to batch size
    axis_grids[1] = axis_grids[1].unsqueeze(0) + gaze_y.unsqueeze(-1).unsqueeze(-1)  # Adjust x-axis with gaze
    axis_grids[2] = axis_grids[2].unsqueeze(0) + gaze_x.unsqueeze(-1).unsqueeze(-1)  # Adjust y-axis with gaz
    # Compute position embeddings for each axis and concatenate
    axis_thetas = [
        get_1d_rotary_embed(axis_dim, axis_grid.flatten(start_dim=1), base)
        for axis_grid in axis_grids
    ]
    thetas = torch.cat(axis_thetas, dim=-1)

    return thetas

class Attention(nn.Module):
    """Multiheaded self-attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        is_causal=False,
        causal_block=1,
        use_sdpa=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.causal_block = causal_block

        if is_causal and causal_block > 1:
            print("Disabling torch spda kernel for block causal attention")
            self.use_sdpa = False
        else:
            self.use_sdpa = use_sdpa

        if not self.use_sdpa:
            self.causal_block_mat = nn.Parameter(
                torch.ones((causal_block, causal_block)).bool(),
                requires_grad=False,
            )

    def forward(self, x, pos_embed=None, attn_mask=None):
        B, N, D = x.shape

        # Attention mask has shape (B, N, N) and dtype torch.bool where a
        # value of True indicates that the element should take part in attention.
        if attn_mask is not None:
            assert len(attn_mask.shape) == 3
            attn_mask = attn_mask.unsqueeze(1)

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        if pos_embed is not None:
            q = apply_rotary_embed(q, pos_embed)
            k = apply_rotary_embed(k, pos_embed)

        if self.use_sdpa:
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p=self.attn_drop.p, is_causal=self.is_causal
            )
        else:
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            if self.is_causal:
                assert attn_mask is None
                assert N % self.causal_block == 0
                num_blocks = N // self.causal_block
                block_diag_mat = torch.block_diag(
                    *[self.causal_block_mat for _ in range(num_blocks)]
                )
                triu_mat = torch.triu(
                    torch.ones(N, N, device=x.device), diagonal=1
                ).bool()
                mask = torch.logical_and(~block_diag_mat, triu_mat)
                attn = attn.masked_fill(mask.view(1, 1, N, N), float("-inf"))
            if attn_mask is not None:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    """Multiheaded cross-attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_spda=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_spda = use_spda

    def forward(self, x, c, x_pos_embed=None, c_pos_embed=None):
        B, Nx, D = x.shape
        q = (
            self.q(x)
            .reshape(B, Nx, self.num_heads, D // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        if x_pos_embed is not None:
            q = apply_rotary_embed(q, x_pos_embed)

        B, Nc, D = c.shape
        kv = (
            self.kv(c)
            .reshape(B, Nc, 2, self.num_heads, D // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        if c_pos_embed is not None:
            k = apply_rotary_embed(k, c_pos_embed)

        if self.use_spda:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        else:
            xattn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
            xattn = xattn.softmax(dim=-1)
            xattn = self.attn_drop(xattn)
            x = xattn @ v

        x = x.transpose(1, 2).reshape(B, Nx, D)
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

    def forward(self, x, t, c, x_pos_embed=None, c_pos_embed=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos_embed=x_pos_embed)
        x = x + self.xattn(self.norm2(x), c, x_pos_embed=x_pos_embed, c_pos_embed=c_pos_embed)
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

    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
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

    def forward(self, x, timestep, cond, x_pos_embed=None, c_pos_embed=None):
        """
        Forward pass of DiT.
        x: (B, N, D) - Input features, where B is batch size, N is sequence length, and D is feature dimension.
        timestep: (B,) - Timestep indices for the diffusion process.
        cond: (B, D) - Conditioning features, where D is the feature dimension.
        """
        x = self.x_embed(x)  
        t = self.time_embed(timestep)
        for block in self.blocks:
            x = block(x, t, cond, x_pos_embed=x_pos_embed, c_pos_embed=c_pos_embed)
        x = self.final_layer(x, t)
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

    def forward(self, x, c=None, x_pos_embed=None, c_pos_embed=None):
        """
        x: [B, S, D] - Query input
        context: [B, S_ctx, D] - Key/Value input
        mask: Optional attention mask
        """
        x = x + self.attn(self.norm1(x), pos_embed=x_pos_embed)
        if self.use_xattn:
            assert c is not None, "Cross-attention requires conditioning context"
            x = x + self.xattn(self.norm2(x), c, x_pos_embed=x_pos_embed, c_pos_embed=c_pos_embed)
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

    def forward(self, x, pos_embed=None):
        for layer in self.layers:
            x = layer(x, x_pos_embed=pos_embed)
        return x
    
class AttentionPooling(nn.Module):
    """
    Attention pooling layer that uses DiT as the backbone.
    """
    def __init__(self, hidden_size, num_queries, depth=2, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_queries = num_queries
        self.query_tokens = nn.Embedding(num_queries, hidden_size)
        self.register_buffer("x_pos_embed", get_1d_rotary_embed(hidden_size // num_heads, torch.arange(num_queries)))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.blocks = nn.ModuleList([
            AttentionBlock(
                use_xattn=True,
                hidden_size=hidden_size,
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                attn_drop=dropout,
                proj_drop=dropout,
            ) for i in range(depth)
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

    def forward(self, c, c_pos_embed=None):
        """
        context: (B, N, D) - Input features, where B is batch size, N is sequence length, and D is feature dimension.
        """
        B = c.size(0)
        x = self.query_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        c = self.norm1(c)
        for block in self.blocks:
            x = block(x=x, c=c, x_pos_embed=self.x_pos_embed, c_pos_embed=c_pos_embed) 
        x = self.norm2(x)
        return x