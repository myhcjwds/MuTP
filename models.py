from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TKBCModel(nn.Module, ABC):
    @abstractmethod
    def get_rhs(self, chunk_begin: int, chunk_size: int):
        pass

    @abstractmethod
    def get_queries(self, queries: torch.Tensor):
        pass

    @abstractmethod
    def score(self, x: torch.Tensor):
        pass

    def get_ranking(
            self, queries, filters, year2id={},
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of quadruples (lhs, rel, rhs, timestam)
        :param filters: filters[(lhs, rel, ts)] gives the elements to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :param chunk_size: maximum number of candidates processed at once
        :return:
        """
        if chunk_size < 0:
            chunk_size = self.sizes[2]
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            c_begin = 0
            while c_begin < self.sizes[2]:
                b_begin = 0
                rhs = self.get_rhs(c_begin, chunk_size)  # 将输入进来的训练集分为几个batch
                while b_begin < len(queries):
                    if queries.shape[1] > 4:  # time intervals exist   对五元组中时间戳的处理
                        these_queries = queries[b_begin:b_begin + batch_size]
                        start_queries = []
                        end_queries = []
                        for triple in these_queries:
                            if triple[3].split('-')[0] == '####':
                                start_idx = -1
                                start = -5000
                            elif triple[3][0] == '-':
                                start = -int(triple[3].split('-')[1].replace('#', '0'))
                            elif triple[3][0] != '-':
                                start = int(triple[3].split('-')[0].replace('#', '0'))
                            if triple[4].split('-')[0] == '####':
                                end_idx = -1
                                end = 5000
                            elif triple[4][0] == '-':
                                end = -int(triple[4].split('-')[1].replace('#', '0'))
                            elif triple[4][0] != '-':
                                end = int(triple[4].split('-')[0].replace('#', '0'))
                            for key, time_idx in sorted(year2id.items(), key=lambda x: x[1]):  # 时间戳转换成id
                                if start >= key[0] and start <= key[1]:
                                    start_idx = time_idx
                                if end >= key[0] and end <= key[1]:
                                    end_idx = time_idx

                            if start_idx < 0:
                                start_queries.append(
                                    [int(triple[0]), int(triple[1]) + self.sizes[1] // 4, int(triple[2]), end_idx])
                            else:
                                start_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            if end_idx < 0:
                                end_queries.append([int(triple[0]), int(triple[1]), int(triple[2]), start_idx])
                            else:
                                end_queries.append(
                                    [int(triple[0]), int(triple[1]) + self.sizes[1] // 4, int(triple[2]), end_idx])

                        start_queries = torch.from_numpy(np.array(start_queries).astype('int64')).cuda()
                        end_queries = torch.from_numpy(np.array(end_queries).astype('int64')).cuda()

                        q_s = self.get_queries(start_queries)
                        q_e = self.get_queries(end_queries)
                        scores = q_s @ rhs + q_e @ rhs
                        targets = self.score(start_queries) + self.score(end_queries)
                    else:
                        these_queries = queries[b_begin:b_begin + batch_size]  # 500, 4
                        q = self.get_queries(these_queries)  # 500, 400
                        """
                        if use_left_queries:
                            lhs_queries = torch.ones(these_queries.size()).long().cuda()
                            lhs_queries[:,1] = (these_queries[:,1]+self.sizes[1]//2)%self.sizes[1]
                            lhs_queries[:,0] = these_queries[:,2]
                            lhs_queries[:,2] = these_queries[:,0]
                            lhs_queries[:,3] = these_queries[:,3]
                            q_lhs = self.get_lhs_queries(lhs_queries)

                            scores = q @ rhs +  q_lhs @ rhs
                            targets = self.score(these_queries) + self.score(lhs_queries)
                        """

                        scores = q @ rhs
                        targets = self.score(these_queries)

                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"

                    # set filtered and true scores to -1e6 to be ignored
                    # take care that scores are chunked
                    for i, query in enumerate(these_queries):
                        if queries.shape[1] > 4:
                            filter_out = filters[int(query[0]), int(query[1]), query[3], query[4]]
                            filter_out += [int(queries[b_begin + i, 2])]
                        else:
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                            filter_out += [queries[b_begin + i, 2].item()]
                        if chunk_size < self.sizes[2]:
                            filter_in_chunk = [
                                int(x - c_begin) for x in filter_out
                                if c_begin <= x < c_begin + chunk_size
                            ]
                            scores[i, torch.LongTensor(filter_in_chunk)] = -1e6
                        else:
                            scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()

                    b_begin += batch_size

                c_begin += chunk_size
        return ranks


class MuTP(TKBCModel):

    def __init__(self, sizes: Tuple[int, int, int, int], rank: int,
                 no_time_emb: bool = False, init_size: float = 1e-2,
                 n_scales: int = 8):
        super(MuTP, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.n_scales = n_scales

        self.W = nn.Embedding(2 * rank, 1, sparse=True)
        self.W.weight.data *= 0

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in [sizes[0], sizes[1], sizes[3]]  # without no_time_emb
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        n_entities, n_relations, _, n_timestamps = sizes
        self.n_timestamps = n_timestamps

        self.time_base_emb = nn.Embedding(n_timestamps, rank)
        self.time_base_emb.weight.data *= init_size

        base_inv_freq = 1.0 / (10000 ** (torch.arange(0, rank, 2).float() / rank))
        self.register_buffer("base_inv_freq", base_inv_freq)

        self.scale_freq_factors = nn.Parameter(
            torch.linspace(2.0, 0.5, n_scales)
        )

        self.rel_scale_weights = nn.Embedding(n_relations, n_scales)
        nn.init.zeros_(self.rel_scale_weights.weight)

        self.no_time_emb = no_time_emb
        self.pi = 3.14159265358979323846

    @staticmethod
    def has_time():
        return True

    def _rope_time_scale(self, time_ids: torch.Tensor, inv_freq: torch.Tensor):
        time_base = self.time_base_emb(time_ids)  # [B, rank]
        dim = time_base.shape[-1]

        if dim % 2 == 1:
            time_base = F.pad(time_base, (0, 1), mode='constant', value=0.0)

        pos = time_ids.float().unsqueeze(1)  # [B, 1]
        freqs = pos * inv_freq.unsqueeze(0)  # [B, rank/2]
        sin, cos = freqs.sin(), freqs.cos()

        x_even = time_base[..., 0::2]
        x_odd = time_base[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        time_rot = torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)  # [B, padded_rank]
        if dim % 2 == 1:
            time_rot = time_rot[..., :dim]  # remove padding

        return time_rot

    def _time_multi_scale(self, rel_ids: torch.Tensor, time_ids: torch.Tensor):
        time_scales_list = []
        
        for scale_idx in range(self.n_scales):
            freq_factor = self.scale_freq_factors[scale_idx]
            inv_freq = self.base_inv_freq / freq_factor
            time_rot = self._rope_time_scale(time_ids, inv_freq)
            time_scales_list.append(time_rot)
        time_scales = torch.stack(time_scales_list, dim=1)  # [B, n_scales, rank]
        
        scale_logits = self.rel_scale_weights(rel_ids)  # [B, n_scales]
        scale_weights = torch.softmax(scale_logits, dim=-1).unsqueeze(-1)  # [B, n_scales, 1]
        
        return time_scales, scale_weights

    def score(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        time = time[:, :self.rank], time[:, self.rank:]

        time_scales, scale_weights = self._time_multi_scale(x[:, 1], x[:, 3])

        rel_expanded = rel[0].unsqueeze(1)
        time_phase_expanded = time[1].unsqueeze(1)

        rt_all = (rel_expanded + time_scales) * time_phase_expanded
        rt_0 = (scale_weights * rt_all).sum(dim=1)

        rt = rt_0, rel[1]
        return torch.sum(
            ((lhs[0] + rt[1]) * rt[0]) * rhs[0], 1, keepdim=True
        )

    def forward(self, x):

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        time = self.embeddings[2](x[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        time_scales, scale_weights = self._time_multi_scale(x[:, 1], x[:, 3])

        rel_expanded = rel[0].unsqueeze(1)  # [B, rank]--->[B, 1, rank]
        time_phase_expanded = time[1].unsqueeze(1)  # [B, rank]--->[B, 1, rank]
        rt_all = (rel_expanded + time_scales) * time_phase_expanded  # [B, n_scales, rank]
        rt_0 = (scale_weights * rt_all).sum(dim=1)  # [B, rank]

        right = self.embeddings[0].weight
        right = right[:, :self.rank], right[:, self.rank:]

        rt = rt_0, rel[1]

        return (
                       ((lhs[0] + rt[1]) * rt[0]) @ right[0].t()
               ), (
                   torch.sqrt(lhs[0] ** 2),
                   torch.sqrt(rt[0] ** 2 + rt[1] ** 2),
                   torch.sqrt(rhs[0] ** 2)
               ), self.embeddings[2].weight[:-1] if self.no_time_emb else self.embeddings[2].weight, scale_weights

    def get_rhs(self, chunk_begin: int, chunk_size: int):
        return self.embeddings[0].weight.data[chunk_begin:chunk_begin + chunk_size][:, :self.rank].transpose(0, 1)

    def get_queries(self, queries: torch.Tensor):
        lhs = self.embeddings[0](queries[:, 0])
        rel = self.embeddings[1](queries[:, 1])
        time = self.embeddings[2](queries[:, 3])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank] / (1 / self.pi), rel[:, self.rank:] / (1 / self.pi)
        time = time[:, :self.rank], time[:, self.rank:]

        time_scales, scale_weights = self._time_multi_scale(queries[:, 1], queries[:, 3])

        rel_expanded = rel[0].unsqueeze(1)  # [B, 1, rank]
        time_phase_expanded = time[1].unsqueeze(1)  # [B, 1, rank]
        rt_all = (rel_expanded + time_scales) * time_phase_expanded  # [B, n_scales, rank]
        rt_0 = (scale_weights * rt_all).sum(dim=1)  # [B, rank]

        rt = rt_0, rel[1]
        return (lhs[0] + rt[1]) * rt[0]

