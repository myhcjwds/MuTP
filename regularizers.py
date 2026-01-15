from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for f in factors:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / factors[0].shape[0]


class Lambda3(Regularizer):
    def __init__(self, weight: float):
        super(Lambda3, self).__init__()
        self.weight = weight

    def forward(self, factor):
        ddiff = factor[1:] - factor[:-1]
        rank = int(ddiff.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)

class Linear3(Regularizer):
    def __init__(self, weight: float):
        super(Linear3, self).__init__()
        self.weight = weight

    def forward(self, factor, W):
        rank = int(factor.shape[1] / 2)
        ddiff = factor[1:] - factor[:-1] - W.weight[:rank*2].t()
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class Spiral3(Regularizer):
    def __init__(self, weight: float):
        super(Spiral3, self).__init__()
        self.weight = weight

    def forward(self, factor, time_phase):
        ddiff = factor[1:] - factor[:-1] 
        ddiff_pahse = time_phase[1:] - time_phase[:-1]
        rank = int(ddiff.shape[1] / 2)
        rank1 = int(ddiff_pahse.shape[1] / 2)
        diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2 + ddiff_pahse[:, :rank1]**2 + ddiff_pahse[:, rank1:]**2)**3
        return self.weight * torch.sum(diff) / (factor.shape[0] - 1)


class ScaleRegularizer(Regularizer):
    def __init__(self):
        super(ScaleRegularizer, self).__init__()
    
    def forward(self, scale_weights: torch.Tensor, rel_scale_weights: torch.nn.Parameter):
        reg = 0.0
        
        # 1. Entropy regularization
        scale_weights_flat = scale_weights.squeeze(-1)  # [B, n_scales]
        n_scales = scale_weights_flat.shape[-1]
        
        if n_scales > 1:
            entropy = -torch.sum(scale_weights_flat * torch.log(scale_weights_flat + 1e-10), dim=-1)
            max_entropy = torch.log(torch.tensor(n_scales, dtype=torch.float32, device=scale_weights_flat.device))
            if max_entropy > 1e-10:
                entropy_loss = torch.mean(entropy / max_entropy)
                if not (torch.isnan(entropy_loss) or torch.isinf(entropy_loss)):
                    reg += entropy_loss

        # 2. L3 regularization on parameters
        rel_scale_weights_clamped = torch.clamp(rel_scale_weights, min=-50, max=50)
        param_loss = torch.sum(torch.abs(rel_scale_weights_clamped) ** 3) / rel_scale_weights.shape[0]
        
        if not (torch.isnan(param_loss) or torch.isinf(param_loss)):
            reg += param_loss
        
        return reg
