import torch
from dataclasses import dataclass
from torch import nn

@dataclass
class TestPredictorOutput:
    loss: torch.Tensor
    logits: torch.Tensor

class BaseTestPredictor(nn.Module):
    def __init__(self, tokenizer_id):
        super().__init__()
        self.tokenizer_id = tokenizer_id