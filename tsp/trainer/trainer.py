import os

import tqdm
from tsp.models.codebert import CodeBertTestPredictor
from tsp.szz_dataset.szz import get_dataloaders
from dataclasses import dataclass
import transformers
from torch import nn, optim
import torch

def batch_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {item[0]: batch_to(item[1], device) for item in batch.items()}
    else:
        raise Exception()

@dataclass
class TrainerConfig:
    epochs: int = 20
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lr: int = 1e-5
    data_path: str = './tsp/szz_dataset/sample_data.json'
    codebert_model_id: str = 'codistai/codeBERT-small-v2'
    experiment_name: str = 'default'
    eval_steps: int = 100

class Trainer:
    def __init__(self, config=None, device=0):
        if config is None:
            config = TrainerConfig()
        self.device = device
        self.config = config
        
        self.epochs = 0
        self.steps = 0
        self.micro_steps = 0
        
        self.init_dataloader()
        self.init_model()
        self.init_optimizer()
    
    def init_model(self):
        self.model = CodeBertTestPredictor().to(self.device).train()
        for m in self.model.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = True
    
    def init_dataloader(self):
        self.train_loader, self.valid_loader = get_dataloaders(
            path=self.config.data_path,
            batch_size=self.config.batch_size,
            tokenizer=transformers.AutoTokenizer.from_pretrained(self.config.codebert_model_id),
        )
    
    def init_optimizer(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def checkpoint_path(self):
        os.makedirs(f'./saves/trainer/{self.config.experiment_name}', exist_ok=True)
        return os.path.join(f'./saves/trainer/{self.config.experiment_name}', 'checkpoint.pth')
    
    def load(self, path=None):
        if path is None: path = self.checkpoint_path()
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        del state
        print('loaded', path)
    
    def save(self, path=None):
        if path is None: path = self.checkpoint_path()
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, path)
        print('saved', path)
    
    def train_epoch(self):
        self.model.train()
        
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for batch in pbar:
                batch = batch_to(batch, self.device)
                
                with torch.autocast('cuda', torch.float16):
                    output = self.model(**batch)
                loss = output.loss
                
                self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
                self.micro_steps += 1
                
                if (self.micro_steps % self.config.gradient_accumulation_steps) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.steps += 1
                    
                    if (self.steps % self.config.eval_steps) == 0:
                        self.evaluate()
                        self.model.train()
                
                pbar.set_description(f'[{self.epochs+1}/{self.config.epochs}] L:{loss:.6f}')
    
    def evaluate(self):
        self.model.eval()
        
        loss_sum = 0
        acc_sum = 0
        count = 0
        for batch in tqdm.tqdm(self.valid_loader, dynamic_ncols=True):
            batch = batch_to(batch, self.device)
            
            with torch.no_grad(), torch.autocast('cuda', torch.float16):
                output = self.model(**batch)
            loss = output.loss
            loss_sum += loss.item() * len(batch['input_ids'])
            
            label = batch['labels']
            acc_sum += (torch.argmax(output.logits, dim=-1, keepdim=False).indices == label).float().sum().item()
            count += len(batch['input_ids'])
        
        return {
            'loss': loss_sum / count,
            'accuracy': acc_sum / count,
        }
    
    def main(self):
        accuracies = []
        for i in range(self.config.epochs):
            self.epochs = i
            self.train_epoch()
            acc = self.evaluate()
            accuracies.append(acc)
        return accuracies

if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()