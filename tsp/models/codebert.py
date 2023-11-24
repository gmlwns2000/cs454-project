from cProfile import label
from dataclasses import dataclass
from typing import Optional
from torch import logit, nn, relu
import torch
import transformers
from tsp.models.modeling_roberta import RobertaModel

@dataclass
class TestPredictorOutput:
    loss: torch.Tensor
    logits: torch.Tensor

class CodeBertTestPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.bert = transformers.AutoModel.from_pretrained('codistai/codeBERT-small-v2')
        self.bert = RobertaModel.from_pretrained('codistai/codeBERT-small-v2')
        lm_hidden_size = self.bert.config.hidden_size
        
        self.past_commit_state_encoder = nn.Linear(3, 16)
        self.past_commit_encoder_cls_token = nn.Parameter(torch.randn((1, 1, lm_hidden_size+16)))
        self.past_commit_encoder = nn.LSTM(lm_hidden_size+16, 128, 1, batch_first=True)
        
        self.current_commit_encoder = nn.Sequential(
            nn.Linear(lm_hidden_size, 128),
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
        N, WIND, PTOK = past_commit_input_ids.shape
        assert past_commit_states.shape == (N, WIND, 3)
        
        # encode each past commit's prompts
        p_input_ids = past_commit_input_ids.view(N*WIND, PTOK)
        p_masks = past_commit_attention_masks.view(N*WIND, PTOK)
        p_bert_output = self.bert(
            input_ids=p_input_ids,
            attention_mask=p_masks
        )
        p_bert_output = p_bert_output.last_hidden_state[:, 0, :]
        p_bert_output = p_bert_output.view(N, WIND, -1)
        
        # encode each past commit's states (0: not bug, 1: bug, 2: unknown)
        p_states = self.past_commit_state_encoder(past_commit_states)
        
        # construct input of sequential encoder
        p_commits = torch.cat([p_bert_output, p_states], dim=-1)
        p_commits = torch.cat([
            p_commits, 
            self.past_commit_encoder_cls_token.expand(N, 1, -1)
        ], dim=1)
        
        # perform lstm to encode p_commits into single vector
        p_encodings, _lstm_state = self.past_commit_encoder(p_commits)
        p_encodings = p_encodings[:, -1, :]
        
        # encode current commit's prompts
        c_bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        c_bert_output = c_bert_output.last_hidden_state[:, 0, :]
        
        # encode bert output into fixed size vector
        c_encodings = self.current_commit_encoder(c_bert_output)
        
        encodings = torch.cat([p_encodings, c_encodings], dim=-1)
        logits = self.classifier(encodings)
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                input=logits,
                target=labels,
            )
        
        return TestPredictorOutput(
            loss=loss,
            logits=logits
        )