from torch import nn, Tensor
import torch
from diffusers.training_utils import EMAModel
from huggingface_hub import PyTorchModelHubMixin
import abc

class Policy(nn.Module, PyTorchModelHubMixin, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_scheduler(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> torch.optim.lr_scheduler.LambdaLR | None:
        """Return the learning rate scheduler to be used for training.

        If no scheduler is needed, return None.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def get_ema(self) -> EMAModel | None:
        """Return the EMA model to be used for training.

        If no EMA is needed, return None.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_delta_timestamps(self) -> dict[str, list[int]]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError
    