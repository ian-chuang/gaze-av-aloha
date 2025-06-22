import torch
from torch import Tensor, nn

# create enum for norm mode either MEAN_STD or MIN_MAX 
class NormalizationMode:
    MEAN_STD = "mean_std"
    MIN_MAX = "min_max"

def create_stats_buffers(
    key_norm_mode: dict[str, str],
    stats: dict[str, dict[str, Tensor]],
) -> dict[str, dict[str, nn.ParameterDict]]:
    stats_buffers = {}
    for key, stat in stats.items():
        if key not in key_norm_mode:
            continue
        if key_norm_mode[key] == NormalizationMode.MEAN_STD:
            mean = torch.tensor(stat["mean"], dtype=torch.float32) 
            std = torch.tensor(stat["std"], dtype=torch.float32) 
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif key_norm_mode[key] == NormalizationMode.MIN_MAX:
            min_val = torch.tensor(stat["min"], dtype=torch.float32)
            max_val = torch.tensor(stat["max"], dtype=torch.float32)
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min_val, requires_grad=False),
                    "max": nn.Parameter(max_val, requires_grad=False),
                }
            )
        else:
            raise ValueError(f"Unknown normalization mode: {key_norm_mode[key]}")
        stats_buffers[key] = buffer
    return stats_buffers

class Normalize(nn.Module):
    def __init__(
        self,
        key_norm_mode: list[str],
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.key_norm_mode = key_norm_mode
        self.stats = stats
        stats_buffers = create_stats_buffers(key_norm_mode, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key in self.key_norm_mode:
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            if self.key_norm_mode[key] == NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif self.key_norm_mode[key] == NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                batch[key] = (batch[key] - min) / (max - min + 1e-8)
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(f"Unknown normalization mode: {self.key_norm_mode[key]}")
        return batch

class Unnormalize(nn.Module):
    def __init__(
        self,
        key_norm_mode: list[str],
        stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__()
        self.key_norm_mode = key_norm_mode
        self.stats = stats
        stats_buffers = create_stats_buffers(key_norm_mode, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key in self.key_norm_mode:
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            if self.key_norm_mode[key] == NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                batch[key] = batch[key] * std + mean
            elif self.key_norm_mode[key] == NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(f"Unknown normalization mode: {self.key_norm_mode[key]}")
        return batch
