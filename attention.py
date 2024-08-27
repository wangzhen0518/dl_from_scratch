import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from typing import Union, Optional
import math


def merge_mask(attention_mask: Tensor, key_padding_mask: Tensor, target_type: Optional[torch.dtype] = torch.float):
    match (attention_mask is not None, key_padding_mask is not None):
        case (False, False):
            mask = None
        case (True, False):
            mask = attention_mask
        case (False, True):
            bs, s = key_padding_mask.size()
            mask = key_padding_mask.reshape(bs, 1, 1, s)
            if mask.dtype == torch.bool:
                # TODO 模型不一定是 torch.float 类型，需要增加可修改类型的部分
                mask = torch.zeros_like(mask, dtype=target_type).masked_fill_(mask, value=-1e8)
        case (True, True):
            bs, _, _, s = attention_mask.size()
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(attention_mask, dtype=target_type).masked_fill_(attention_mask, value=-1e8)
            if not torch.is_floating_point(attention_mask):
                raise ValueError(f"only bool and floating types of masks are supported, but input is {attention_mask.dtype}")

            key_padding_mask = key_padding_mask.reshape(bs, 1, 1, s)
            if key_padding_mask.dtype == torch.bool:
                key_padding_mask = torch.zeros_like(key_padding_mask, dtype=target_type).masked_fill_(key_padding_mask, value=-1e8)
            if not torch.is_floating_point(key_padding_mask):
                raise ValueError(f"only bool and floating types of masks are supported, but input is {key_padding_mask.dtype}")

            mask = attention_mask + key_padding_mask
    return mask


def cal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    key_padding_mask: Optional[Tensor] = None,
    dropout_p: float = 0.1,
) -> Tensor:
    embed_dim = query.size(-1)
    score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(embed_dim)
    mask = merge_mask(attention_mask, key_padding_mask, torch.float)
    if mask is not None:
        score += mask
    attention = F.softmax(score, dim=-1)
    if dropout_p > 0.0:
        attention = F.dropout(attention, dropout_p)
    x = torch.matmul(attention, value)
    return x, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, dropout_p: float = 0.1) -> None:
        super().__init__()
        assert embed_dim % nhead == 0

        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.dropout_p = dropout_p

        self.w_q = nn.Linear(embed_dim, embed_dim)
        self.w_k = nn.Linear(embed_dim, embed_dim)
        self.w_v = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bs = len(query)
        (Q, K, V) = (
            linear(x).reshape(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
            for linear, x in zip((self.w_q, self.w_k, self.w_v), (query, key, value))
        )
        x, attention = cal_attention(Q, K, V, attention_mask, key_padding_mask, self.dropout_p)
        x = x.transpose(1, 2).reshape(bs, -1, self.embed_dim)
        x = self.w_o(x)
        return x, attention


class MultiQueryAttention(nn.Module):
    def __init__(self, embed_size: int, nhead: int, dropout_p: float) -> None:
        super().__init__()

        assert embed_size % nhead == 0
        self.embed_size = embed_size
        self.nhead = nhead
        self.head_dim = embed_size // nhead
        self.dropout_p = dropout_p

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, self.head_dim)
        self.w_v = nn.Linear(embed_size, self.head_dim)
        self.w_o = nn.Linear(embed_size, embed_size)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bs = len(query)
        Q = self.w_q(query).reshape(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
        (K, V) = (linear(x).reshape(bs, -1, 1, self.head_dim).transpose(1, 2) for linear, x in zip((self.w_k, self.w_v), (key, value)))
        x, attention = cal_attention(Q, K, V, attention_mask, key_padding_mask, self.dropout_p)
        x = x.transpose(1, 2).reshape(bs, -1, self.embed_size)
        x = self.w_o(x)
        return x, attention


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_size: int, nhead: int, ngroup: int, dropout_p: float) -> None:
        super().__init__()

        assert embed_size % nhead == 0
        assert nhead % ngroup == 0
        self.embed_size = embed_size
        self.nhead = nhead
        self.ngroup = ngroup
        self.head_dim = embed_size // nhead
        self.group_head = nhead // ngroup
        self.dropout_p = dropout_p

        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, ngroup * self.head_dim)
        self.w_v = nn.Linear(embed_size, ngroup * self.head_dim)
        self.w_o = nn.Linear(embed_size, embed_size)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        bs = len(query)
        Q = self.w_q(query).reshape(bs, -1, self.nhead, self.head_dim).transpose(1, 2)
        (K, V) = (
            linear(x).reshape(bs, -1, self.ngroup, self.head_dim).transpose(1, 2).repeat_interleave(self.group_head, dim=1)
            for linear, x in zip((self.w_k, self.w_v), (key, value))
        )
        x, attention = cal_attention(Q, K, V, attention_mask, key_padding_mask, self.dropout_p)
        x = x.transpose(1, 2).reshape(bs, -1, self.embed_size)
        x = self.w_o(x)
        return x, attention
