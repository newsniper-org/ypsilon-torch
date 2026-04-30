from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from typing import final, Protocol
from typing_extensions import override

from ypsilon_torch import FPDTypeIdx, get_float_dtype

class ActivationFunctionBase(nn.Module, ABC):
    """Abstract base for activation functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def assert_tensor(cls, tensor: torch.Tensor) -> bool:
        raise NotImplementedError()

    def forward_asserted(self, input: torch.Tensor) -> torch.Tensor:
        assert self.__class__.assert_tensor(input)
        output: torch.Tensor = self.forward(input)
        assert self.__class__.assert_tensor(output)
        return output

class RealActivationFunctionBase(ActivationFunctionBase, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @override
    def assert_tensor(cls, tensor: torch.Tensor) -> bool:
        return not tensor.dtype.is_complex

class ComplexActivationFunctionBase(ActivationFunctionBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    @override
    def assert_tensor(cls, tensor: torch.Tensor) -> bool:
        return tensor.dtype.is_complex

class BiasedComplexActivationFunctionBase(ComplexActivationFunctionBase, ABC):
    """Abstract base for complex-valued activation functions."""

    @abstractmethod
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64):
        super().__init__()

        self.bias: nn.Parameter = nn.Parameter(torch.zeros(features, dtype=get_float_dtype(dtype_idx)))
        self.eps: float = eps