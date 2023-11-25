import os
import argparse
from dataclasses import asdict, dataclass

import tqdm
import wandb
import torch
from torch import nn, optim
import transformers

from src.szz_dataset.szz import get_dataloaders, DatasetConfig
from src import models

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
    gradient_checkpointing: bool = True
    lr: int = 1e-5
    data_path: str = './src/szz_dataset/sample_data.json'
    model_id: str = 'codebert_test_predictor'
    experiment_name: str = 'default'
    eval_steps: int = 1000
    wandb_project_name: str = 'cs454'
    dataset_config: DatasetConfig = DatasetConfig()

class Trainer:
    def __init__(self, config=None, device=0):
        if config is None:
            config = TrainerConfig()
        self.device = device
        self.config = config
        
        self.epochs = 0
        self.steps = 0
        self.micro_steps = 0
        
        self.init_model()
        self.init_dataloader()
        self.init_optimizer()
    
    def init_model(self):
        model_id = self.config.model_id
        if models.has_model(model_id):
            self.model = models.get_model(model_id)
        else:
            raise Exception(f'Given model id `{model_id}` is not found. Defined models: {models.list_model()}. Did you register model properly?')
        
        self.model = self.model.to(self.device).train()
        for m in self.model.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = self.config.gradient_checkpointing
    
    def init_dataloader(self):
        self.train_loader, self.valid_loader, self.valid_unseen_project = get_dataloaders(
            path=self.config.data_path,
            batch_size=self.config.batch_size,
            tokenizer=transformers.AutoTokenizer.from_pretrained(self.model.tokenizer_id),
            config=self.config.dataset_config
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
        
        loss_sum = 0
        
        with tqdm.tqdm(self.train_loader, dynamic_ncols=True) as pbar:
            for ibatch, batch in enumerate(pbar):
                batch = batch_to(batch, self.device)
                
                with torch.autocast('cuda', torch.float16):
                    output = self.model(**batch)
                loss = output.loss
                
                self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
                loss_sum += (loss / self.config.gradient_accumulation_steps).item()
                self.micro_steps += 1
                
                if (self.micro_steps % self.config.gradient_accumulation_steps) == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.steps += 1
                    
                    wandb.log({
                        'train/loss': loss_sum,
                        'train/epoch': ibatch / len(pbar) + self.epochs,
                    }, step=self.steps)
                    
                    if (self.steps % self.config.eval_steps) == 0:
                        result = self.evaluate()
                        result_wandb = {f'eval/{item[0]}': item[1] for item in result[0].items()}
                        result_wandb.update({f'eval/{item[0]}': item[1] for item in result[1].items()})
                        wandb.log(result_wandb, step=self.steps)
                        
                        self.save()
                        self.model.train()
                    
                    loss_sum = 0
                
                pbar.set_description(f'[{self.epochs+1}/{self.config.epochs}] L:{loss:.6f}')
    
    def evaluate_subset(self, subset='valid'):
        self.model.eval()
        
        loader = None
        if subset == 'train':
            loader = self.train_loader
        elif subset == 'valid':
            loader = self.valid_loader
        elif subset == 'valid_unseen_project':
            loader = self.valid_unseen_project
        else:
            raise Exception()
        
        loss_sum = 0
        acc_sum = 0
        count = 0
        for batch in tqdm.tqdm(loader, dynamic_ncols=True, desc=subset):
            batch = batch_to(batch, self.device)
            
            with torch.no_grad(), torch.autocast('cuda', torch.float16):
                output = self.model(**batch)
            loss = output.loss
            loss_sum += loss.item() * len(batch['input_ids'])
            
            label = batch['labels']
            acc_sum += ((torch.argmax(output.logits, dim=-1, keepdim=False) == label) * 1.0).sum().item()
            count += len(batch['input_ids'])
        
        result = {
            'loss': loss_sum / count,
            'accuracy': acc_sum / count,
        }
        
        return result
    
    def evaluate(self):
        result_valid = self.evaluate_subset('valid')
        result_valid_unseen_project = self.evaluate_subset('valid_unseen_project')
        print(
            f'epoch={self.epochs}, steps={self.steps} (micro_steps={self.micro_steps}) | '
            f'result@valid={result_valid}, result@unseen_valid={result_valid_unseen_project}'
        )
        return result_valid, result_valid_unseen_project
    
    def main(self):
        # self.evaluate()
        
        wandb.init(
            project=self.config.wandb_project_name,
            config=asdict(self.config),
        )
        
        accuracies = []
        for i in range(self.config.epochs):
            self.epochs = i
            
            self.train_epoch()
            
            acc = self.evaluate()
            self.save()
            
            result_wandb = {f'eval/valid_{item[0]}': item[1] for item in acc[0].items()}
            result_wandb.update({f'eval/valid_unseen_{item[0]}': item[1] for item in acc[1].items()})
            wandb.log(result_wandb, step=self.steps)
            
            accuracies.append(acc)
        
        print('-'*40)
        print('- Summary')
        print('-'*40)
        
        for i in range(len(accuracies)):
            print(f'epoch {i+1}', accuracies[i])
        
        return accuracies

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        type=str, 
        default='codebert_test_predictor', 
        choices=models.list_model()
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./src/szz_dataset/sample_data.json'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='default',
    )
    parser.add_argument(
        '--allow_oracle_past_state',
        default=False,
        action='store_true'
    )
    parser.add_argument(
        '--window_size',
        default=40,
        type=int,
    )
    
    args = parser.parse_args()
    
    config = TrainerConfig(
        model_id=args.model,
        data_path=args.data_path,
        experiment_name=args.experiment_name,
        dataset_config=DatasetConfig(
            window_size=args.window_size,
            allow_oracle_past_state=args.allow_oracle_past_state
        )
    )
    trainer = Trainer(config=config)
    trainer.main()