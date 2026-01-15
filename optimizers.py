import tqdm
import torch
from torch import nn
from torch import optim

from models import TKBCModel
from regularizers import Regularizer
from datasets import TemporalDataset

class TKBCOptimizer(object):
    def __init__(
            self, model: TKBCModel,
            emb_regularizer: Regularizer, temporal_regularizer: Regularizer,
            optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True,
            device: torch.device = None,
            scale_regularizer: Regularizer = None,
            scale_reg_weight: float = 0.0
    ):
        self.model = model
        self.emb_regularizer = emb_regularizer
        self.temporal_regularizer = temporal_regularizer
        self.scale_regularizer = scale_regularizer
        self.scale_reg_weight = scale_reg_weight
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device if device is not None else next(model.parameters()).device

    def epoch(self, examples: torch.LongTensor):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')
        total_batches = max(1, (examples.shape[0] + self.batch_size - 1) // self.batch_size)
        total_loss = total_reg = total_time = total_scale = 0.0

        with tqdm.tqdm(total=total_batches, unit='batch', disable=not self.verbose) as bar:
            bar.set_description('train')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()

                predictions, factors, time, scale_weights = self.model.forward(input_batch)
                time_input = time
                if isinstance(time, tuple):
                    time_input = time[0]
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.emb_regularizer.forward(factors)
                l_time = torch.zeros_like(l_reg)
                if time_input is not None:
                    l_time = self.temporal_regularizer.forward(time_input)

                # Multi-scale regularization: combine entropy and L3 losses, controlled by single weight
                l_scale = torch.zeros_like(l_reg)
                if self.scale_regularizer is not None and self.scale_reg_weight > 0:
                    rel_scale_weights = self.model.rel_scale_weights.weight
                    l_scale = self.scale_reg_weight * self.scale_regularizer.forward(scale_weights, rel_scale_weights)

                l = l_fit + l_reg + l_time + l_scale

                self.optimizer.zero_grad()
                l.backward()
                for param in self.model.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                            param.grad = torch.nan_to_num(param.grad)

                self.optimizer.step()

                total_loss += l_fit.item()
                total_reg += l_reg.item()
                total_time += l_time.item()
                total_scale += l_scale.item()

                b_begin += self.batch_size
                bar.update(1)
                bar.set_postfix(
                    loss=f'{l_fit.item():.4f}',
                    reg=f'{l_reg.item():.4f}',
                    cont=f'{l_time.item():.4f}',
                    scale=f'{l_scale.item():.4f}'
                )

        avg_loss = total_loss / total_batches
        avg_reg = total_reg / total_batches
        avg_time = total_time / total_batches
        avg_scale = total_scale / total_batches

        return {'loss': avg_loss, 'reg': avg_reg, 'cont': avg_time, 'scale': avg_scale}


