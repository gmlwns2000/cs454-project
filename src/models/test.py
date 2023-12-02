"""
Lookup previous commit by attention, using current commit as query.
"""

from dataclasses import dataclass
from typing import Optional
from torch import logit, nn, relu
import torch
import transformers
import torch.nn.functional as F
from modules.modeling_roberta import RobertaModel
from base_test_predictor import BaseTestPredictor, TestPredictorOutput
from registry import register
import sys

class CodeBertAttenTestPredictorLong(BaseTestPredictor):
    def __init__(
        self,
        hidden_size=768,
        state_hidden_size=32,
        lstm_n_layers=1,
    ):
        super().__init__(tokenizer_id='codistai/codeBERT-small-v2')
    
        
        self.bert = RobertaModel.from_pretrained('codistai/codeBERT-small-v2')
        self.lm_hidden_size = self.bert.config.hidden_size
        self.num_heads = self.bert.config.num_attention_heads
        
        self.p_bert_output_class_token = nn.Parameter(torch.randn(1, 1, self.lm_hidden_size))
        self.c_bert_output_class_token = nn.Parameter(torch.randn(1, 1, self.lm_hidden_size))
        
        self.p_bert_query = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        self.p_bert_key = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        self.p_bert_value = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        
        self.c_bert_query = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        self.c_bert_key = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        self.c_bert_value = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        
        self.past_commit_state_encoder = nn.Linear(2, state_hidden_size)
        p_hidden_state = self.lm_hidden_size + state_hidden_size
        self.past_commit_state_key = nn.Linear(p_hidden_state, self.lm_hidden_size)
        self.past_commit_state_value = nn.Linear(p_hidden_state, self.lm_hidden_size)
        
        self.current_commit_query = nn.Sequential(
            nn.Linear(self.lm_hidden_size, self.lm_hidden_size),
        )
        self.current_commit_pool = nn.Linear(self.lm_hidden_size, self.lm_hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.lm_hidden_size+self.lm_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
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
        assert past_commit_states.shape == (N, WIND, 2)
        
        # encode each past commit's prompts
        p_input_ids = past_commit_input_ids.view(N*WIND, PTOK)
        p_masks = past_commit_attention_masks.view(N*WIND, PTOK)
        
        p_bert_output_list = []
        
        # append class token to the p_bert_output_list
        p_bert_output_seg_class_token = self.p_bert_output_class_token.repeat(N*WIND,1,1)
        p_bert_output_list.append(p_bert_output_seg_class_token)
        
        # accumulate p_bert_output_seg
        for i in range(0,PTOK,self.lm_hidden_size):
            p_input_ids_seg = p_input_ids[:,i:i+self.lm_hidden_size]
            p_attention_mask_seg = p_masks[:,i:i+self.lm_hidden_size]
            p_bert_output_seg = self.bert(
                input_ids=p_input_ids_seg,
                attention_mask=p_attention_mask_seg
            )
            p_bert_output_seg = p_bert_output_seg.last_hidden_state[:, 0, :]
            p_bert_output_seg = p_bert_output_seg.unsqueeze(1)
            p_bert_output_list.append(p_bert_output_seg)
        
        # concatenate p_bert_output_segs 
        p_bert_output = torch.cat(p_bert_output_list, dim=1)
        
        p_bert_output_query = self.p_bert_query(p_bert_output)
        p_bert_output_key = self.p_bert_key(p_bert_output)
        p_bert_output_value = self.p_bert_value(p_bert_output)

        H = self.num_heads
        HEAD_DIM = p_bert_output_key.shape[-1] // H
        N_WIND, SEG_NUM, _ = p_bert_output_key.shape
        p_bert_output_query = p_bert_output_query.view(N_WIND, SEG_NUM, H, HEAD_DIM).permute(0, 2, 1, 3)
        p_bert_output_key = p_bert_output_key.view(N_WIND, SEG_NUM, H, HEAD_DIM).permute(0, 2, 1, 3)
        p_bert_output_value = p_bert_output_value.view(N_WIND, SEG_NUM, H, HEAD_DIM).permute(0, 2, 1, 3)
        p_bert_output_context = F.scaled_dot_product_attention(
            query=p_bert_output_query,
            key=p_bert_output_key,
            value=p_bert_output_value,
            dropout_p=0.1,
        )
        p_bert_output_context = p_bert_output_context.permute(0, 2, 1, 3).reshape(N_WIND, SEG_NUM, H * HEAD_DIM)[:,0,:]
        p_bert_output_context = p_bert_output_context.reshape(N, WIND, -1)
        
        # encode each past commit's states (0: not bug, 1: bug, 2: unknown)
        p_states = self.past_commit_state_encoder(past_commit_states)
        
        # construct input of sequential encoder
        p_commits = torch.cat([p_bert_output_context, p_states], dim=-1)
        
        c_bert_output_list = []
        
        # append class token to the c_bert_output_list
        c_bert_output_seg_class_token = self.c_bert_output_class_token.repeat(N,1,1)
        c_bert_output_list.append(c_bert_output_seg_class_token)
        
        for i in range(0,PTOK,self.lm_hidden_size):
            c_input_ids_seg = input_ids[:,i:i+self.lm_hidden_size]
            c_attention_mask_seg = attention_mask[:,i:i+self.lm_hidden_size]
            c_bert_output_seg = self.bert(
                input_ids=c_input_ids_seg,
                attention_mask=c_attention_mask_seg
            )
            c_bert_output_seg = c_bert_output_seg.last_hidden_state[:, 0, :]
            c_bert_output_seg = c_bert_output_seg.unsqueeze(1)
            c_bert_output_list.append(c_bert_output_seg)
            
        c_bert_output = torch.cat(c_bert_output_list, dim=1)
        
        c_bert_output_query = self.c_bert_query(c_bert_output)
        c_bert_output_key = self.c_bert_key(c_bert_output)
        c_bert_output_value = self.c_bert_value(c_bert_output)
        
        H = self.num_heads
        HEAD_DIM = c_bert_output_key.shape[-1] // H
        N, SEG_NUM, _ = c_bert_output_key.shape
        c_bert_output_query = c_bert_output_query.view(N, SEG_NUM, H, HEAD_DIM).permute(0, 2, 1, 3)
        c_bert_output_key = c_bert_output_key.view(N, SEG_NUM, H, HEAD_DIM).permute(0, 2, 1, 3)
        c_bert_output_value = c_bert_output_value.view(N, SEG_NUM, H, HEAD_DIM).permute(0, 2, 1, 3)
        c_bert_output_context = F.scaled_dot_product_attention(
            query=c_bert_output_query,
            key=c_bert_output_key,
            value=c_bert_output_value,
            dropout_p=0.1,
        )
        
        c_bert_output_context = c_bert_output_context.permute(0, 2, 1, 3).reshape(N, SEG_NUM, H * HEAD_DIM)[:,0,:]
        c_bert_output_context = c_bert_output_context.reshape(N, 1, -1)
        
        # now perform attention
        c_query = self.current_commit_query(c_bert_output_context)
        p_key = self.past_commit_state_key(p_commits)
        p_value = self.past_commit_state_value(p_commits)
        H = self.num_heads
        HEAD_DIM = p_key.shape[-1] // H
        N, WIND, _ = p_key.shape
        p_key = p_key.view(N, WIND, H, HEAD_DIM).permute(0, 2, 1, 3)
        p_value = p_value.view(N, WIND, H, HEAD_DIM).permute(0, 2, 1, 3)
        c_query = c_query.view(N, 1, H, HEAD_DIM).permute(0, 2, 1, 3)
        context = F.scaled_dot_product_attention(
            query=c_query,
            key=p_key,
            value=p_value,
            dropout_p=0.1,
        )
        context = context.permute(0, 2, 1, 3).reshape(N, H * HEAD_DIM)
        
        # calcuate logits using pooled current commit encoding
        c_pool = torch.cat([c_bert_output, context], dim=-1)
        logits = self.classifier(c_pool)
        
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

@register('codebert_atten_long')
def codebert_atten_long():
    return CodeBertAttenTestPredictorLong()

# past_commit_states = torch.randint(2,(1,5,2)).type(torch.float)
# past_commit_input_ids = torch.randint(10,(1,5,2044))
# past_commit_attention_masks = torch.randint(10,(1,5,2044))
# input_ids = torch.randint(10, (1,4088))
# attention_mask = torch.randint(10, (1,4088))
# labels = torch.randint(2,(1,2))

# model = CodeBertAttenTestPredictorLong()

# model(past_commit_states,past_commit_input_ids,past_commit_attention_masks,input_ids,attention_mask,labels)