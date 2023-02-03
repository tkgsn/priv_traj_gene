from torch.utils.data import Dataset
import torch
import random
import numpy as np


class TrajectoryDataset(Dataset):
    #Init dataset
    def __init__(self, data, window_size, seq_len, n_bins, dataset_name="dataset", random_mask=False, mask_start=1):
        dataset = self
        
        dataset.data = data
        dataset.n_locations = (n_bins+2)**2
        dataset.vocab = list(range(dataset.n_locations)) + ['<start>', '<ignore>', '<oov>', '<mask>', '<cls>']
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.seq_len = seq_len
        dataset.mask_start = mask_start
        dataset.random_mask = random_mask
        dataset.dataset_name = dataset_name
        dataset.window_size = window_size
        dataset.training_seq_len = dataset.window_size + dataset.seq_len
        
        print("random_mask:", dataset.random_mask)
        #special tags
        dataset.START_IDX = dataset.vocab['<start>']
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
        dataset.CLS_IDX = dataset.vocab['<cls>']
        dataset.padded_data = np.concatenate([[[dataset.START_IDX]*dataset.window_size]*len(dataset.data),dataset.data], axis=1)
    
    def __str__(self):
        return self.dataset_name
        
    #fetch data
    # def __getitem__(self, index, p_random_mask=0.15):
    def __getitem__(self, index):
        dataset = self
        
        
        start_position = np.random.choice(range(1,dataset.seq_len))
        s = dataset.padded_data[index, start_position:start_position+dataset.window_size]
        target = dataset.padded_data[index, start_position+dataset.window_size]

        return {'input': torch.Tensor(s).long(),
                'target': torch.Tensor([target]).long()}

    def __len__(self):
        return len(self.data)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.data[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s] 
        return s
    
    
    
class RealFakeDataset(Dataset):
    
    def __init__(self, real_data, fake_data):
        self.real_data = real_data
        self.num_real_data = len(real_data)
        self.fake_data = fake_data
        self.num_fake_data = len(fake_data)
        
        
    def __getitem__(self, index):
        num_real_data = self.num_real_data
        num_fake_data = self.num_fake_data
        
        if index >= self.num_real_data:
            target = 0
            input = self.fake_data[index-self.num_real_data]
        else:
            target = 1
            input = self.real_data[index]
            
        
        return input, target
        
    def __len__(self):
        return self.num_real_data + self.num_fake_data