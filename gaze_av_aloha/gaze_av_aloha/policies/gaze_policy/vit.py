from typing import Optional, Type, Callable
import torch
from timm.layers import Mlp
from torch import nn, Tensor
from huggingface_hub import PyTorchModelHubMixin

def named_apply(fn: Callable, module: torch.nn.Module, name="", depth_first=True, include_root=False) -> torch.nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module

def init_weights_vit_timm(module: torch.nn.Module, name: str = ""):
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

def extend_valid_token_mask(
    valid_token_mask: Optional[torch.Tensor], num_registers: int
) -> Optional[torch.Tensor]:
    """
    Extends the valid token mask to include the register tokens.

    Parameters:
    - valid_token_mask (B, N): The original valid token mask.

    Returns:
    - extended_valid_token_mask (B, N + R): The extended valid token mask.
    """
    if valid_token_mask is None:
        return None
    return torch.cat(
        [
            torch.ones(
                valid_token_mask.shape[0],
                num_registers,
                dtype=torch.bool,
                device=valid_token_mask.device,
            ),
            valid_token_mask,
        ],
        dim=1,
    )

def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Block(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        act_layer: Type[torch.nn.Module],
        mlp_dim: Optional[int] = None,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        if mlp_dim is None:
            mlp_dim = 4 * dim

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop
        )
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(
        self, x: torch.Tensor, invalid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        x = self.attn(x, x, x, key_padding_mask=invalid_token_mask)[0]
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ImageEncoder(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_tokens: int,
        num_registers: int,
        patch_size: int,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        act_layer: str,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()

        act_layer = get_activation_fn(act_layer)

        self.patch_emb = torch.nn.Linear(patch_size * patch_size * 3, embedding_dim)

        self.pos_enc = torch.nn.Parameter(torch.randn(num_tokens, embedding_dim))

        self.reg_tokens = torch.nn.Parameter(torch.randn(num_registers, embedding_dim))

        self.blocks = torch.nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                Block(
                    dim=embedding_dim,
                    num_heads=num_heads,
                    act_layer=act_layer,
                    drop=drop,
                    drop_path=drop_path,
                )
            )

        self.init_weights()

    @property
    def embed_dim(self) -> int:
        return self.patch_emb.out_features

    def init_weights(self):
        torch.nn.init.trunc_normal_(self.pos_enc, std=0.02)
        torch.nn.init.normal_(self.reg_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def forward(
        self, foveated_tokens: torch.Tensor, valid_token_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters:
        - foveated_tokens (B, N, E): The input foveated image tokens.
        - valid_token_mask (B, N): An optional mask indicating which tokens are valid.

        Returns:
        - features (B, N, C): The output feature vectors, one per input token
        - register_features (B, N, C): The register feature vectors, one per register token
        """
        embedded_tokens = self.patch_emb(
            foveated_tokens.flatten(2)
        ) + self.pos_enc.unsqueeze(0)

        num_registers = self.reg_tokens.shape[0]

        embedded_tokens = torch.cat(
            [
                self.reg_tokens.unsqueeze(0).expand(embedded_tokens.shape[0], -1, -1),
                embedded_tokens,
            ],
            dim=1,
        )

        valid_token_mask = extend_valid_token_mask(valid_token_mask, num_registers)

        invalid_token_mask = (
            valid_token_mask.logical_not() if valid_token_mask is not None else None
        )

        transformed_tokens = embedded_tokens
        for block in self.blocks:
            transformed_tokens = block(transformed_tokens, invalid_token_mask)

        features = transformed_tokens #self.neck(transformed_tokens)

        return features[:, num_registers:], features[:, :num_registers] # return features, register_features

def create_vit_s(num_tokens, patch_size) -> ImageEncoder:
    return ImageEncoder(
        num_tokens=num_tokens,
        num_registers=1,
        patch_size=patch_size,
        depth=12,
        embedding_dim=384,
        num_heads=6,
        act_layer="gelu",
        drop=0.0,
        drop_path=0.1,
    )

def create_vit_b(num_tokens, patch_size) -> ImageEncoder:
    return ImageEncoder(
        num_tokens=num_tokens,
        num_registers=1,
        patch_size=patch_size,
        depth=12,
        embedding_dim=768,
        num_heads=12,
        act_layer="gelu",
        drop=0.0,
        drop_path=0.1,
    )

def get_activation_fn(name: str) -> Type[torch.nn.Module]:
    """
    Returns the activation function class based on the provided name.
    """
    if name == "gelu":
        return torch.nn.GELU
    elif name == "relu":
        return torch.nn.ReLU
    else:
        raise ValueError(f"Unsupported activation function: {name}")
