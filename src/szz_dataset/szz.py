import json
import os
from typing import List, Literal
from dataclasses import dataclass, field

import tqdm
import torch
from torch.utils.data import Dataset, random_split, DataLoader
import transformers
import torch.nn.functional as F

@dataclass
class DatasetConfig:
    window_size: int = 40
    valid_ratio: float = 0.05
    allow_oracle_past_state: bool = False
    past_commit_filter: Literal['none', 'only_buggy'] = 'none'

def collate_input_ids(input_ids: List[torch.Tensor], pad_token_id: int):
    max_len = max([len(i) for i in input_ids])
    output_ids = torch.empty((len(input_ids), max_len), dtype=input_ids[0].dtype)
    output_ids.fill_(pad_token_id)
    output_masks = torch.zeros_like(output_ids, dtype=torch.float32)
    for i in range(len(input_ids)):
        j = len(input_ids[i])
        output_ids[i, :j] = input_ids[i]
        output_masks[i, :j] = 1
    return output_ids, output_masks

class SZZDataset(Dataset):
    def __init__(
        self, 
        path, 
        tokenizer: transformers.AutoTokenizer, 
        max_seq_len: int = 1022, 
        project_split = 'train', 
        project_split_num_valid = 1, 
        config = None
    ):
        if config is None:
            config = DatasetConfig()
        self.config = config
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        self.path = path
        self.window = config.window_size
        
        with open(path, 'r') as f:
            data = json.load(f)
        # put number of entries in front of project name, so we can sort the project ascending order of size.
        self.data = {}
        for k, v in data.items():
            self.data[f'{str(len(v)).zfill(9)}_{k}'] = v
            assert len(v) <= 999999999
        
        self.projects = list(sorted(list(self.data.keys())))
        projects_train, projects_valid = self.projects[project_split_num_valid:], self.projects[:project_split_num_valid]
        if project_split == 'train':
            self.projects = projects_train
        elif project_split == 'valid':
            self.projects = projects_valid
        else:
            raise Exception()
        
        self.count = 0
        self.index_to_project = []
        self.index_to_base = []
        for project in self.projects:
            self.index_to_base += [self.count, ] * len(self.data[project])
            self.index_to_project += [project, ] * len(self.data[project])
            self.count += len(self.data[project])
            project_data = self.data[project]
            cache_path = path + f'.{project}.pth'
            if os.path.exists(cache_path) and os.environ.get('RESET_CACHE', '0') == '0':
                input_ids = torch.load(cache_path)
            else:
                input_ids = []
                for i in tqdm.tqdm(range(len(project_data)), dynamic_ncols=True, leave=False, desc=f'{project}'):
                    sample = project_data[i]
                    input_ids.append(tokenizer(sample['text'], truncation=True, max_length=max_seq_len, return_tensors='pt', return_attention_mask=False).input_ids[0])
                torch.save(input_ids, cache_path)
            for i in range(len(project_data)):
                project_data[i]['input_ids'] = input_ids[i]
        self.pad_token_id = tokenizer.pad_token_id
    
    def __len__(self):
        return self.count
    
    def __getitem__(self, index):
        """
        returns
            past_commit_states: Tensor(shape=(WINDOW, 2))
            past_commit_input_ids: Tensor(shape=(WINDOW, P_TOK))
            past_commit_attention_masks: Tensor(shape=(WINDOW, P_TOK))
            input_ids: Tensor(shape=(TOK,))
            attention_mask: Tensor(shape=(TOK,)),
            label: Tensor(shape=(1,))
        """
        
        project = self.index_to_project[index]
        index = index - self.index_to_base[index]
        
        past_commits = []
        if self.config.past_commit_filter == 'none':
            for i in range(self.window):
                j = max(index - self.window + i, 0)
                past_commit = self.data[project][j]
                past_commits.append(past_commit)
        elif self.config.past_commit_filter == 'only_buggy':
            j = max(index - 1, 0)
            while len(past_commits) < self.window:
                if j == 0:
                    past_commits.append(self.data[project][j])
                else:
                    item = self.data[project][j]
                    if item['is_buggy'] > 0.5:
                        past_commits.append(item)
                    j -= 1
                    j = max(j, 0)
        else:
            raise Exception(self.config.past_commit_filter)
        
        past_commit_states = []
        past_commit_input_ids = []
        for past_commit in past_commits:
            if past_commit['reported_index'] < index or self.config.allow_oracle_past_state:
                past_commit_states.append(int(past_commit['is_buggy']))
            else:
                past_commit_states.append(0)
            past_commit_input_ids.append(past_commit['input_ids'])
        
        past_commit_input_ids, past_commit_attention_masks = collate_input_ids(past_commit_input_ids, pad_token_id=self.pad_token_id)
        past_commit_states_pt = torch.zeros((len(past_commit_states), 2))
        past_commit_states_pt.scatter_(
            dim=1, 
            index=torch.tensor(past_commit_states, dtype=torch.int64).unsqueeze(-1), 
            value=1
        )
        past_commit_states = past_commit_states_pt
        
        current_commit = self.data[project][index]
        input_ids = current_commit['input_ids']
        label = torch.tensor(current_commit['is_buggy'])
        
        return {
            'past_commit_states': past_commit_states,
            'past_commit_input_ids': past_commit_input_ids,
            'past_commit_attention_masks': past_commit_attention_masks,
            'input_ids': input_ids,
            'label': label
        }

def collate_fn(pad_token_id, items):
    past_commit_states = torch.stack([item['past_commit_states'] for item in items], dim=0)
    past_commit_input_ids_max_len = max([item['past_commit_input_ids'].shape[-1] for item in items])
    past_commit_input_ids = torch.stack([
        F.pad(
            item['past_commit_input_ids'].unsqueeze(0), 
            pad=(0, past_commit_input_ids_max_len-item['past_commit_input_ids'].shape[-1]), 
            mode='constant', 
            value=pad_token_id
        ).squeeze(0) 
        for item in items
    ])
    past_commit_attention_masks = torch.stack([
        F.pad(
            item['past_commit_attention_masks'].unsqueeze(0), 
            pad=(0, past_commit_input_ids_max_len-item['past_commit_attention_masks'].shape[-1]), 
            mode='constant', 
            value=0
        ).squeeze(0) 
        for item in items
    ])
    input_ids, attention_mask = collate_input_ids([item['input_ids'] for item in items], pad_token_id=pad_token_id)
    labels = torch.tensor([item['label'] for item in items])
    
    return {
        'past_commit_states': past_commit_states,
        'past_commit_input_ids': past_commit_input_ids,
        'past_commit_attention_masks': past_commit_attention_masks,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }

def get_dataloaders(path, batch_size, tokenizer, config=None, max_seq_len=1022, seed=42, num_workers=4):
    if config is None:
        config = DatasetConfig()
    ds = SZZDataset(path, tokenizer, max_seq_len, config=config)
    generator = torch.Generator().manual_seed(seed)
    train_ds, valid_ds = random_split(ds, [1-config.valid_ratio, config.valid_ratio], generator=generator)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=lambda x: collate_fn(ds.pad_token_id, x), num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size, shuffle=False, collate_fn=lambda x: collate_fn(ds.pad_token_id, x), num_workers=num_workers)
    
    valid_unseen_project_ds = SZZDataset(path, tokenizer, max_seq_len, project_split='valid', config=config)
    valid_unseen_project_loader = DataLoader(valid_unseen_project_ds, batch_size, shuffle=False, collate_fn=lambda x: collate_fn(ds.pad_token_id, x), num_workers=num_workers)
    
    return train_loader, valid_loader, valid_unseen_project_loader

if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('codistai/codeBERT-small-v2')
    train, valid = get_dataloaders('./src/szz_dataset/sample_data.json', 4, tokenizer)
    print(len(train), len(valid))
    for batch in tqdm.tqdm(train):
        # print(batch)
        # input()
        # print(batch['input_ids'].shape)
        pass
    for batch in tqdm.tqdm(valid):
        # print(batch)
        # input()
        # print(batch['input_ids'].shape)
        pass