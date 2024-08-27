from torch import nn
import torch
from typing import List, Union, Tuple, Optional
from numbers import Integral
from torch import Tensor


class MyLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        if isinstance(normalized_shape, Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.has_bias = bias
        self.device = device
        self.dtype = dtype
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(normalized_shape, device=device, dtype=dtype)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(normalized_shape, device=device, dtype=dtype)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _check_shape(self, size: List[int]) -> Tuple[int]:
        """Check the input tensor's shape and return the dimensions that need to be normed."""
        n = len(size)
        mean_n = len(self.normalized_shape)
        if n < mean_n:
            raise ValueError(
                f"Input tensor's dimension {size} is smaller than {self.num_features} that needs norm"
            )
        delta = n - mean_n
        for idx in reversed(range(delta, n)):
            in_di = size[idx]
            b_di = self.normalized_shape[idx - delta]
            if in_di != b_di:
                raise ValueError(
                    f"Shape inconsistency: input tensor({in_di}) != target shape({b_di}) in dimension {idx}."
                )
        return tuple(range(delta, n))

    def forward(self, x: Tensor) -> Tensor:
        dim = self._check_shape(x.size())
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)
        output = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            output = output * self.weight
            if self.has_bias:
                output = output + self.bias
        return output


@torch.no_grad()
def is_same(x: Tensor, y: Tensor):
    x, y = x.squeeze(), y.squeeze()
    assert x.shape == y.shape
    diff = (x - y).abs().max().item()
    return diff < 1e-5, diff


def norm(x: Tensor, dim: Union[int, List[int], Tuple[int]]):
    return (x - x.mean(dim=dim, keepdim=True)) / torch.sqrt(
        x.var(dim=dim, keepdim=True, unbiased=False) + 1e-5
    )


ln = nn.LayerNorm((4, 5))
x = torch.randn(3, 4, 5)
y = ln(x)
print(y)

mln = MyLayerNorm((4, 5))
mx = x.clone()
my = mln(mx)

print(my)
