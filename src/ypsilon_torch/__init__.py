import warnings

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import final, Literal, Generic, TypedDict, Unpack, TypeVar, Protocol, Self

try:
    from typing import override
except:
    from typing_extensions import override

FPDTypeIdx = Literal[32, 64, 128]

EXPERIMENTAL_DTYPE_IDXS: set[FPDTypeIdx] = {32}

COMPLEX_DTYPES_DICT: dict[FPDTypeIdx, torch.dtype] = {
    32: torch.complex32,
    64: torch.complex64,
    128: torch.complex128,
}

FLOAT_DTYPES_DICT: dict[FPDTypeIdx, torch.dtype] = {
    32: torch.float16,
    64: torch.float32,
    128: torch.float64,
}


def get_complex_dtype(dtype_idx: FPDTypeIdx) -> torch.dtype:
    """Return a complex dtype for the given dtype index.
        - 32 -> complex32
        - 64 -> complex64
        - 128 -> complex128
    """
    return COMPLEX_DTYPES_DICT[dtype_idx]

def get_float_dtype(dtype_idx: FPDTypeIdx) -> torch.dtype:
    """Return a complex dtype for the given dtype index.
        - 32 -> float16
        - 64 -> float32
        - 128 -> float64
    """
    return FLOAT_DTYPES_DICT[dtype_idx]

class NonLearnableProcessorBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """실제 변환 로직을 구현하는 추상 메서드"""
        pass

    def is_valid_input(self, x: torch.Tensor) -> bool:
        """
        입력 텐서의 유효성을 검사하는 메서드.
        기본적으로 True를 반환하며, 필요시 하위 클래스에서 오버라이드.
        """
        return True

    @final
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        PyTorch 모듈의 forward 메서드.
        입력 검증 후 transform을 호출합니다.
        """
        if not self.is_valid_input(x):
            raise ValueError(f"Invalid input shape or dtype: {x.shape}, {x.dtype}")
        return self.transform(x)


class NonLearnableSynchronizedProcessorBase[C: TypedDict](nn.Module, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, *xs: torch.Tensor, **caches: Unpack[C]) -> tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def is_valid_input(self, x: torch.Tensor) -> bool:
        """
        입력 텐서의 유효성을 검사하는 메서드.
        기본적으로 True를 반환하며, 필요시 하위 클래스에서 오버라이드.
        """
        return True

    @final
    def forward(self, *xs: torch.Tensor, **caches: Unpack[C]) -> tuple[torch.Tensor, ...]:
        """
        PyTorch 모듈의 forward 메서드.
        입력 검증 후 transform을 호출합니다.
        """
        for x in xs:
            if not self.is_valid_input(x):
                raise ValueError(f"Invalid input shape or dtype: {x.shape}, {x.dtype}")
            else:
                continue
        else:
            return self.transform(*xs, **caches)