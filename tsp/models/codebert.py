from typing import Optional
from torch import nn, relu
import torch
import transformers

class CodeBertTestPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.bert = transformers.AutoModel.from_pretrained('codistai/codeBERT-small-v2')
        
        self.past_commit_state_encoder = nn.Linear(3, 16)
        self.past_commit_encoder = nn.LSTM(768+16, 128, 1, batch_first=True)
        
        self.now_commit_encoder = nn.Sequential(
            nn.Linear(768, 128),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128+128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
    
    def forward(
        self, 
        past_commit_states: torch.Tensor,
        past_commit_input_ids: torch.Tensor,
        past_commit_attention_masks: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass