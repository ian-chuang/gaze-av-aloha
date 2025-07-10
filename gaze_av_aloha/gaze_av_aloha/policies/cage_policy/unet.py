import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from einops.layers.torch import Rearrange

class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 3, 2, 1) # used to be 4, 2, 1

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


# legacy resnet block (norm last)
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        Parameters:
            x: (B, C, L)
            cond: (B, D)

        Returns:
            out: (B, C, L)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# Updated ResNetBlock with norm first, to be used with TransformerBlock
class ResNetBlock1D(nn.Module):
    def __init__(self, 
                 dim_in, dim_out, 
                 cond_dim=None,
                 dropout=0.,
                 kernel_size=3,
                 num_groups=8,
                 scale_shift=True):
        super().__init__()

        self.act = nn.Mish()
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.GroupNorm(num_groups, dim_in)
        self.conv1 = nn.Conv1d(dim_in, dim_out, kernel_size, padding=kernel_size // 2)

        self.norm2 = nn.GroupNorm(num_groups, dim_out)
        self.conv2 = nn.Conv1d(dim_out, dim_out, kernel_size, padding=kernel_size // 2)

        self.cond_proj = None
        if cond_dim is not None:
            self.cond_type = 'scale_shift' if scale_shift else 'add'
            self.cond_proj = nn.Linear(cond_dim, dim_out * 2 if scale_shift else dim_out)

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x, cond=None):
        """
        Parameters:
            x: (B, C, L)
            cond: (B, D)
        
        Returns:
            out: (B, C, L)
        """
        out = self.norm1(x)
        out = self.act(out)
        out = self.conv1(out)

        if self.cond_proj is not None:
            cond_emb = self.act(cond)
            cond_emb = self.cond_proj(cond_emb).unsqueeze(-1)

            if self.cond_type == 'scale_shift':
                out = self.norm2(out)

                scale, shift = cond_emb.chunk(2, dim=1)
                out = out * (1+scale) + shift
            else:
                out = out + cond_emb
                out = self.norm2(out)
        else:
            out = self.norm2(out)

        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + self.residual_conv(x)
        return out


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


class TransformerBlock1D(TransformerBlock):
    def forward(self, x, cond=None, mask=None, cond_mask=None):
        x = x.permute(0, 2, 1)              # channel last (B, C, L) -> (B, L, D)
        x = super().forward(x, cond, mask, cond_mask)
        x = x.permute(0, 2, 1).contiguous() # swap back

        return x


# Below are the 1D versions of the blocks used in the UNet of SD
class AttnDownBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim,
                 num_layers=1, add_downsample=True,
                 kernel_size=3, num_groups=8,
                 num_heads=8):
        super().__init__()

        resnets = []
        attentions = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=cond_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
            attentions.append(
                TransformerBlock1D(
                    dim=output_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.downsampler = Downsample1d(output_dim) if add_downsample else nn.Identity()

    def forward(self, x, t_emb, cond):
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x, cond)
        
        return self.downsampler(x), x


class DownBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, timestep_dim,
                 num_layers=1, add_downsample=True,
                 kernel_size=3, num_groups=8):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=timestep_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsampler = Downsample1d(output_dim) if add_downsample else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        for resnet in self.resnets:
            x = resnet(x, t_emb)
        
        return self.downsampler(x), x


class AttnMidBlock1D(nn.Module):
    def __init__(self, dim, cond_dim,
                 num_layers=1,
                 kernel_size=3, num_groups=8,
                 num_heads=8):
        super().__init__()

        # there is always at least one resnet
        resnets = [
            ResNetBlock1D(
                dim, dim,
                cond_dim=cond_dim, 
                kernel_size=kernel_size,
                num_groups=num_groups,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            resnets.append(
                ResNetBlock1D(
                    dim, dim,
                    cond_dim=cond_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
            attentions.append(
                TransformerBlock1D(
                    dim=dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, x, t_emb, cond):
        x = self.resnets[0](x, t_emb)
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            x = attn(x, cond)
            x = resnet(x, t_emb)
        
        return x


class AttnUpBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim,
                 num_layers=1, add_upsample=True,
                 kernel_size=3, num_groups=8,
                 num_heads=8):
        super().__init__()

        resnets = []
        attentions = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=cond_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
            attentions.append(
                TransformerBlock1D(
                    dim=output_dim,
                    cond_dim=cond_dim,
                    num_heads=num_heads,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.upsampler = Upsample1d(output_dim) if add_upsample else nn.Identity()

    def forward(self, x, t_emb, cond):
        for resnet, attn in zip(self.resnets, self.attentions):
            x = resnet(x, t_emb)
            x = attn(x, cond)
        
        return self.upsampler(x), x


class UpBlock1D(nn.Module):
    def __init__(self, input_dim, output_dim, timestep_dim,
                 num_layers=1, add_upsample=True,
                 kernel_size=3, num_groups=8):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            dim_in = input_dim if i == 0 else output_dim
            resnets.append(
                ResNetBlock1D(
                    dim_in, output_dim,
                    cond_dim=timestep_dim, 
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                ),
            )
        self.resnets = nn.ModuleList(resnets)

        self.upsampler = Upsample1d(output_dim) if add_upsample else nn.Identity()

    def forward(self, x, t_emb, cond=None):
        for resnet in self.resnets:
            x = resnet(x, t_emb)
        
        return self.upsampler(x), x

# copied from Diffusion Policy, removed local_cond
class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        conv_kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = Timesteps(dsed, flip_sin_to_cos=False, downscale_freq_shift=1)
        diffusion_step_proj = nn.Sequential(
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=conv_kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=conv_kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=conv_kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=conv_kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.diffusion_step_proj = diffusion_step_proj
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample, timestep, cond=None):
        """
        Parameters:
            sample: (B, Ha, Da)
            timestep: (B,) or int, diffusion step
            cond: (B, ...)

        Returns:
            output: (B, Ha, Da)
        """
        sample = sample.permute(0, 2, 1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.diffusion_step_encoder(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        
        global_feature = self.diffusion_step_proj(t_emb)

        if cond is not None:
            cond = cond.flatten(1, -1)  # B, D
            global_feature = torch.cat([
                global_feature, cond
            ], dim=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.permute(0, 2, 1)


# added attn after each residual block
class ConditionalAttnUnet1D(nn.Module):
    def __init__(self, 
                 input_dim,
                 obs_dim,
                 down_dims=[256,512,1024],
                 layers_per_block=2,
                 conv_kernel_size=3,
                 num_norm_groups=8,
                 num_attn_heads=8,
                 ):
        super().__init__()

        self.timestep_emb = Timesteps(obs_dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.timestep_proj = nn.Sequential(
            nn.Linear(obs_dim, obs_dim * 4),
            nn.SiLU(),
            nn.Linear(obs_dim * 4, obs_dim),
        )

        self.cond_norm = nn.LayerNorm(obs_dim)

        self.conv_in = nn.Conv1d(input_dim, down_dims[0], kernel_size=conv_kernel_size, padding=conv_kernel_size // 2)

        all_dims = [down_dims[0]] + down_dims
        in_out_dims = list(zip(all_dims[:-1], all_dims[1:]))

        down_modules = []
        for i, (dim_in, dim_out) in enumerate(in_out_dims):
            if i == len(in_out_dims) - 1:
                down_modules.append(DownBlock1D(
                    dim_in, dim_out, timestep_dim=obs_dim,
                    num_layers=layers_per_block, add_downsample=False,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                ))
            else:
                down_modules.append(AttnDownBlock1D(
                    dim_in, dim_out, cond_dim=obs_dim,
                    num_layers=layers_per_block, add_downsample=True,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                    num_heads=num_attn_heads,
                ))
        self.down_modules = nn.ModuleList(down_modules)

        self.mid_module = AttnMidBlock1D(
            all_dims[-1], cond_dim=obs_dim,
            num_layers=1,
            kernel_size=conv_kernel_size, num_groups=num_norm_groups,
            num_heads=num_attn_heads,
        )

        up_modules = []
        for i, (dim_out, dim_in) in enumerate(reversed(in_out_dims)):
            is_last = i == len(in_out_dims) - 1
            if i == 0:
                up_modules.append(UpBlock1D(
                    dim_in*2, dim_out, timestep_dim=obs_dim,
                    num_layers=layers_per_block, add_upsample=not is_last,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                ))
            else:
                up_modules.append(AttnUpBlock1D(
                    dim_in*2, dim_out, cond_dim=obs_dim,
                    num_layers=layers_per_block, add_upsample=not is_last,
                    kernel_size=conv_kernel_size, num_groups=num_norm_groups,
                    num_heads=num_attn_heads,
                ))
        self.up_modules = nn.ModuleList(up_modules)
        
        self.conv_out = nn.Sequential(
            nn.GroupNorm(num_norm_groups, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], input_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size // 2),
        )

        # initialize weights
        self.apply(self._init_weights)
        self._rescale_blocks()

    def forward(self, sample, timesteps, cond):
        """
        Parameters:
            sample: (B, Ta, Da)
            timestep: (B,) or int, diffusion step
            cond: (B, ..., D)

        Returns:
            output: (B, Ta, Da) predicted noise
        """
        sample = sample.permute(0, 2, 1)

        if not torch.is_tensor(timesteps):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.timestep_emb(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        t_emb = self.timestep_proj(t_emb)

        cond = cond.flatten(1, -2)  # B, L, D
        cond = self.cond_norm(cond)
        
        x = self.conv_in(sample)
        
        h = []
        for block in self.down_modules:
            x = block(x, t_emb, cond)
            h.append(x[1])
            x = x[0]

        x = self.mid_module(x, t_emb, cond)

        for block in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = block(x, t_emb, cond)[0]

        x = self.conv_out(x)
        return x.permute(0, 2, 1)    # B, Ta, Da

    def _init_weights(self, module):
        init_std = 0.02
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _rescale_blocks(self):
        def rescale(module, layer):
            for m in module.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                    m.weight.data.div_(np.sqrt(2.0 * layer))

        for i, module in enumerate(self.down_modules):
            rescale(module, i+1)

        rescale(self.mid_module, len(self.down_modules)+1)

        for i, module in enumerate(reversed(self.up_modules)):
            rescale(module, i+1)