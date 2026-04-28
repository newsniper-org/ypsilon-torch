from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import torch
import torch.nn as nn

from ypsilon_torch import NonLearnableProcessorBase

class InvertibleTransformsBase[I: InvertibleTransformsBase[Self]](NonLearnableProcessorBase, ABC):
    @abstractmethod
    def get_inverse_transform(self) -> I:
        raise NotImplementedError()