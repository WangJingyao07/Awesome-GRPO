
from abc import ABC, abstractmethod
from typing import Dict
import torch

class AlgorithmBase(ABC):
    def __init__(self, engine, tokenizer, **kwargs):
        self.engine = engine
        self.tokenizer = tokenizer
        for k, v in kwargs.items():
            setattr(self, k, v)

    @abstractmethod
    def step(self, batch: Dict) -> torch.Tensor:
        ...
