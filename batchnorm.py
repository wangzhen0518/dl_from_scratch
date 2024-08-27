from numbers import Integral
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


@torch.no_grad()
def is_same(x: Tensor, y: Tensor):
    x, y = x.squeeze(), y.squeeze()
    assert x.shape == y.shape
    diff = (x - y).abs().max().item()
    return diff < 1e-5, diff


def norm(x: Tensor, dim: Union[int, List[int], Tuple[int]]):
    return (x - x.mean(dim=dim, keepdim=True)) / torch.sqrt(x.var(dim=dim, keepdim=True, unbiased=False) + 1e-5)


bn1 = nn.BatchNorm1d(4)
x1 = torch.FloatTensor([[1, 2, 4, 1], [6, 3, 2, 4], [2, 4, 6, 1]])
y1 = bn1(x1)
print(x1, y1)

bn2 = nn.BatchNorm1d(4)
x2 = torch.randn(3, 4, 5)
y2 = bn2(x2)
print(x2, y2)


class MyBatchNorm(nn.Module):
    def __init__(
        self,
        num_features: Union[int, List[int], Tuple[int]],
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if isinstance(num_features, Integral):
            num_features = (num_features,)
        self.num_features = tuple(num_features)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.register_buffer("running_mean", torch.zeros(num_features, device=device, dtype=dtype))
        self.register_buffer("running_var", torch.ones(num_features, device=device, dtype=dtype))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long, device=device))
        self.running_mean: Tensor
        self.running_var: Tensor
        self.num_batches_tracked: Tensor
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(num_features, device=device, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def _check_shape(self, size: List[int]) -> Tuple[int]:
        """Check the input tensor's shape and return the dimensions that need to be normed."""
        bs = size[0]
        if bs <= 1:
            raise ValueError("Input tensor's batch size should not be 1.")
        n = len(size)
        mean_n = len(self.num_features)
        if n < mean_n:
            raise ValueError(f"Input tensor's dimension {size} is smaller than {self.num_features} that needs norm")
        for idx, (in_di, b_di) in enumerate(zip(size[1:], self.num_features)):
            if in_di != b_di:
                raise ValueError(f"Shape inconsistency: input tensor({in_di}) != target shape({b_di}) in dimension {idx+1}.")
        return (0, *range(mean_n + 1, n))

    def forward(self, x: Tensor):
        #! variation 有 2 种计算方式，有偏的计算方式采用 N 作为分母，无偏的计算方式采用 N-1 作为分母
        #! 在进行 norm 时，使用有偏的 variation 计算方式，而更新 running_var 时，采用无偏的 variation 计算方式
        dim = self._check_shape(x.size())
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True)

        self.running_mean = self.running_mean.reshape_as(mean)
        self.running_var = self.running_var.reshape_as(var)

        self.running_mean = self.momentum * mean + (1 - self.momentum) * self.running_mean
        self.running_var = self.momentum * var + (1 - self.momentum) * self.running_var

        output = (x - mean) / torch.sqrt(x.var(dim=dim, keepdim=True, unbiased=False) + self.eps)
        if self.affine:
            output = output * self.weight + self.bias
        return output


mbn1 = MyBatchNorm(4)
mx1 = torch.FloatTensor([[1, 2, 4, 1], [6, 3, 2, 4], [2, 4, 6, 1]])
my1 = mbn1(mx1)
print(mx1, my1)

mbn2 = MyBatchNorm(4)
mx2 = x2.clone()
my2 = mbn2(mx2)
print(mx2, my2)


bn3 = nn.BatchNorm2d(4)
x3 = torch.randn(3, 4, 5, 6)
y3 = bn3(x3)
print(x3, y3)

mbn3 = MyBatchNorm(4)
mx3 = x3.clone()
my3 = mbn3(mx3)
print(mx3, my3)

mbn4 = MyBatchNorm((4, 5, 6))
mx4 = mx3.clone()
my4 = mbn4(mx4)
print(mx4, my4)
