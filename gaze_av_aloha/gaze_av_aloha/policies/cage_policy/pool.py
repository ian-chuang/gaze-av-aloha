import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention

class TransformerBlock(nn.Module):
    def __init__(self, dim, cond_dim=None, num_heads=8, dropout=0., cross_attn_only=False):
        super().__init__()
        self.cross_attn_only = cross_attn_only
        assert not cross_attn_only or cond_dim is not None, 'If only do cross attention, cond_dim must NOT be None!'

        # self attn
        if not cross_attn_only:
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = Attention(
                dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                dropout=dropout,
            )

        # cross attn
        self.attn2 = None
        if cond_dim is not None:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                dim,
                cross_attention_dim=cond_dim,
                heads=num_heads,
                dim_head=dim // num_heads,
                dropout=dropout,
            )

        # feedforward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout)

    def forward(self, x, cond=None, mask=None, cond_mask=None):
        """
        Parameters:
            x: (B, L, D)
            cond: (B, L, D)
            mask: (B, L, L)
            cond_mask: (B, L, L)

        Returns:
            out: (B, L, D)
        """
        if not self.cross_attn_only:
            norm_x = self.norm1(x)
            x = self.attn1(norm_x, attention_mask=mask) + x

        if self.attn2 is not None:
            norm_x = self.norm2(x)
            x = self.attn2(norm_x, cond, attention_mask=cond_mask) + x

        norm_x = self.norm3(x)
        x = self.ff(norm_x) + x

        return x

class AttentionPooling(nn.Module):
    def __init__(self, dim, cond_dim, num_queries=4, layers=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        self.norm1 = nn.LayerNorm(cond_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim, 
                cond_dim=cond_dim, 
                num_heads=num_heads, 
                dropout=dropout,
                cross_attn_only=True if i == 0 else False,
            ) for i in range(layers)
        ])
        self.norm2 = nn.LayerNorm(dim)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, cond):
        x = self.queries.expand(cond.shape[0], -1, -1)
        c = self.norm1(cond)
        for block in self.blocks:
            x = block(x, c)
        x = self.norm2(x)
        return x
    
# class CAGEPooling(nn.Module):
#     def __init__(self, dim, obs_dim, num_queries=1, layers=4, num_heads=8, dropout=0.1):
#         super().__init__()

#         self.latents = nn.Parameter(torch.randn(1, num_queries, dim))
#         self.obs_norm = nn.LayerNorm(obs_dim)
#         self.x_attn = TransformerBlock(dim, cond_dim=obs_dim, dropout=dropout, cross_attn_only=True)
#         self.blocks = nn.ModuleList([
#             TransformerBlock(dim, num_heads=num_heads, dropout=dropout) for _ in range(layers)
#         ])

#     def forward(self, obs_emb):
#         B, L, D = obs_emb.shape
#         obs_emb = self.obs_norm(obs_emb)

#         # mask = torch.ones(T, T, dtype=torch.bool).tril()
#         # mask = mask.unsqueeze(0).expand(B, T, T).to(device=obs_emb.device)
#         # cond_mask = mask.reshape(B, T, T, 1).repeat(1, 1, N, L).reshape(B, T, N*T*L)

#         latents = self.latents.expand(B, -1, -1)
#         latents = self.x_attn(latents, obs_emb) #cond_mask=cond_mask)
        
#         for block in self.blocks:
#             latents = block(latents) #mask=mask)

#         return latents