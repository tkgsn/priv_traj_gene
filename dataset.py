from torch.utils.data import Dataset
import torch
import random
import numpy as np

def traj_to_format(traj):
    # list a set of states in the trajectory
    # i.e., remove the duplicated states
    states = []
    for state in traj:
        if state not in states:
            states.append(state)
    # convert the list of states to a string
    # i.e., convert the list of states to a format
    format = ''
    for state in traj:
        # convert int to alphabet
        format += chr(states.index(state) + 97)
        # format += str(states.index(state))

    return format

def padded_collate(batch):
    # compute max_len
    max_len = max([len(x["input"]) for x in batch])
    inputs = []
    targets = []
    for record in batch:
        s = record["input"]
        target = record["target"]
        inputs.append(s + [dataset.IGNORE_IDX] * (max_len - len(s)))
        targets.append(target + [dataset.IGNORE_IDX] * (max_len - len(target)))

    return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}

def padded_collate_without_end(batch):
    # compute max_len
    max_len = max([len(x["input"]) for x in batch])-1
    if max_len == 0:
        max_len = 1
    inputs = []
    targets = []
    for record in batch:
        # remove elements with length 1
        if len(record["input"]) == 1:
            continue

        s = record["input"][:-1]
        target = record["target"][:-1]
        inputs.append(s + [dataset.IGNORE_IDX] * (max_len - len(s)))
        targets.append(target + [dataset.IGNORE_IDX] * (max_len - len(target)))

    return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}


def padded_collate_without_end_only_first_markov(batch):
    # compute max_len
    inputs = []
    targets = []
    for record in batch:
        if len(record["input"]) < 2:
            continue

        s = record["input"][:-1][:1]
        target = record["target"][:-1][:1]
        inputs.append(s)
        targets.append(target)

    return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long()}


def make_padded_collate(ignore_idx, start_idx, format_to_label):

    def padded_collate(batch):
        # compute max_len
        max_len = max([len(x["trajectory"]) for x in batch])
        max_time = 60*24-1
        inputs = []
        targets = []
        times = []
        target_times = []
        labels = []

        for record in batch:
            trajectory = record["trajectory"]
            time_trajecotry = record["time_trajectory"]

            format = traj_to_format(trajectory)
            label = format_to_label[format]

            input = [start_idx] + trajectory + [ignore_idx] * (max_len - len(trajectory))
            target = input[1:] + [ignore_idx]

            # convert the duplicated state of target to the ignore_idx
            # if the label is "010", then the second 0 is converted to the ignore_idx
            checked_target = ["a"]
            for i in range(1,len(format)):
                if format[i] not in checked_target:
                    checked_target.append(format[i])
                    continue
                target[i] = ignore_idx

            # convert time_input [0,800,1439,...] -> [0, 800/1439, 1439/1439, ...]
            time_input = [v/max_time for v in time_trajecotry] + [ignore_idx] * (max_len - len(time_trajecotry)+1)
            time_target = time_input[1:] + [ignore_idx]

            inputs.append(input)
            targets.append(target)
            times.append(time_input)
            target_times.append(time_target)
            labels.append(label)

        return {"input":torch.Tensor(inputs).long(), "target":torch.Tensor(targets).long(), "time":torch.Tensor(times).float(), "time_target":torch.Tensor(target_times).float(), "label":torch.Tensor(labels).long()}



        # for record in batch:
        #     if len(record["input"]) == 1:
        #         continue
        #     label = record["label"]
        #     s = record["input"][:-1]
        #     target = record["target"][:-1]
        #     time = convert_time_traj_to_time_traj_float(record["time"])[:-1]
        #     target_time = convert_time_traj_to_time_traj_float(record["time_target"][:-1])
        #     if len(s) != len(time):
        #         continue
        #     inputs.append([start_idx] + s + [ignore_idx] * (max_len - len(s)))
        #     targets.append(target + [ignore_idx] * (max_len - len(target)))
        #     times.append(time + [ignore_idx] * (max_len - len(time)))
        #     target_times.append(target_time + [ignore_idx] * (max_len - len(target_time)))
        #     labels.append(label)

    return padded_collate



class TrajectoryDataset(Dataset):
    #Init dataset
    def __init__(self, data, n_bins, dataset_name="dataset"):
        dataset = self
        
        dataset.data = data
        # print(data)
        # nan_place = np.isnan(data)
        # dataset.seq_lens = np.sum(nan_place==False, axis=1)

        # compute max seq len in one line
        dataset.seq_len = max([len(trajectory) for trajectory in data])

        dataset.n_locations = (n_bins+2)**2
        vocab = list(range(dataset.n_locations)) + ['<end>', '<ignore>', '<start>', '<oov>', '<mask>', '<cls>']
        dataset.vocab = {e:i for i, e in enumerate(vocab)} 
        # dataset.seq_len = seq_len
        # dataset.random_mask = random_mask
        dataset.dataset_name = dataset_name
        # dataset.window_size = window_size
        # dataset.training_seq_len = dataset.window_size + dataset.seq_len
        
        # print("random_mask:", dataset.random_mask)
        #special tags
        dataset.START_IDX = dataset.vocab['<start>']
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
        dataset.CLS_IDX = dataset.vocab['<cls>']
        dataset.END_IDX = dataset.vocab['<end>']



        # dataset.padded_data = np.concatenate([[[dataset.START_IDX]*dataset.window_size]*len(dataset.data),dataset.data], axis=1).astype(int)
        # dataset.padded_data = np.concatenate([[[dataset.START_IDX]]*len(dataset.data),dataset.data], axis=1).astype(int)

        # dataset.start_positions = {}
    
    def __str__(self):
        return self.dataset_name
        
    #fetch data
    # def __getitem__(self, index, p_random_mask=0.15):
    def __getitem__(self, index):
        assert self.data_loader is not None, "data_loader should be set to fix the sequence length for each batch"

        dataset = self
        # this_seq_len = len(dataset.data[index])
        # pad dataset by ignore to be the same length
        # s = dataset.data[index] + [dataset.IGNORE_IDX] * (dataset.seq_len - this_seq_len)
        # target = dataset.data[index][1:] + [dataset.END_IDX] + [dataset.IGNORE_IDX] * (dataset.seq_len - this_seq_len)

        s = dataset.data[index]
        target = dataset.data[index][1:] + [dataset.END_IDX]

        # print(s)
        # print(target)

        return {'input': s, 'target': target}
        # return {'input':torch.Tensor(s).long(), 'target':torch.Tensor(target).long()}

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.start_positions = {}

    #get words id
    def get_sentence_idx(self, index):
        dataset = self


# convert minute to float of [0,1]
def int_to_float_of_minute(minute):
    return minute / 1439

# convert real_time_traj to real_time_traj_float
def convert_time_traj_to_time_traj_float(real_time_traj):
    real_time_traj_float = []
    for time_start, _ in real_time_traj:
        real_time_traj_float.append(int_to_float_of_minute(time_start))
    return real_time_traj_float


class TrajectoryDataset_with_Time(Dataset):
    #Init dataset
    def __init__(self, data, time_data, n_bins, format_to_label, dataset_name="dataset"):
        dataset = self
        
        dataset.data = data
        dataset.seq_len = max([len(trajectory) for trajectory in data])

        dataset.time_data = time_data

        dataset.n_locations = (n_bins+2)**2
        vocab = list(range(dataset.n_locations)) + ['<start>', '<ignore>', '<end>', '<oov>', '<mask>', '<cls>']
        dataset.vocab = {e:i for i, e in enumerate(vocab)} 
        dataset.dataset_name = dataset_name
        dataset.format_to_label = format_to_label

        #special tags
        dataset.START_IDX = dataset.vocab['<start>']
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
        dataset.CLS_IDX = dataset.vocab['<cls>']
        dataset.END_IDX = dataset.vocab['<end>']
    
    def __str__(self):
        return self.dataset_name
        
    # fetch data
    def __getitem__(self, index):
        dataset = self
        # s = dataset.data[index]
        # target = dataset.data[index][1:] + [dataset.END_IDX]
        trajectory = dataset.data[index]
        time_trajectory = dataset.time_data[index]
        

        return {'trajectory': trajectory, 'time_trajectory': time_trajectory}
        # return {'input': s, 'target': target, 'time': time, 'time_target': time_target, 'index': index, 'label': label}
        # return {'input':torch.Tensor(s).long(), 'target':torch.Tensor(target).long()}

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.start_positions = {}

    #get words id
    def get_sentence_idx(self, index):
        dataset = self


class TrajectoryDataset_(Dataset):
    #Init dataset
    def __init__(self, data, window_size, seq_len, n_bins, dataset_name="dataset", random_mask=False):
        dataset = self
        
        dataset.data = data
        # nan_place = np.isnan(data)
        # dataset.seq_lens = np.sum(nan_place==False, axis=1)

        dataset.n_locations = (n_bins+2)**2
        dataset.vocab = list(range(dataset.n_locations)) + ['<end>', '<start>', '<ignore>', '<oov>', '<mask>', '<cls>']
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.seq_len = seq_len
        dataset.random_mask = random_mask
        dataset.dataset_name = dataset_name
        # dataset.window_size = window_size
        # dataset.training_seq_len = dataset.window_size + dataset.seq_len
        
        print("random_mask:", dataset.random_mask)
        #special tags
        dataset.START_IDX = dataset.vocab['<start>']
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
        dataset.CLS_IDX = dataset.vocab['<cls>']
        dataset.END_IDX = dataset.vocab['<end>']



        # dataset.padded_data = np.concatenate([[[dataset.START_IDX]*dataset.window_size]*len(dataset.data),dataset.data], axis=1).astype(int)
        # dataset.padded_data = np.concatenate([[[dataset.START_IDX]]*len(dataset.data),dataset.data], axis=1).astype(int)

        dataset.start_positions = {}
    
    def __str__(self):
        return self.dataset_name
        
    #fetch data
    # def __getitem__(self, index, p_random_mask=0.15):
    def __getitem__(self, index):
        assert self.data_loader is not None, "data_loader should be set to fix the sequence length for each batch"

        dataset = self
        
        batch_id = self.data_loader._get_iterator()._num_yielded

        seq_len = dataset.seq_lens[index]
        if batch_id not in self.start_positions:
            start_position = np.random.choice(range(1,seq_len+2))
            self.start_positions[batch_id] = start_position
        else:
            start_position = self.start_positions[batch_id]
        

        # nan_place = np.isnan(dataset.data[index])
        # seq_len = np.sum(nan_place==False)
        # start_position = np.random.choice(range(1,seq_len+1))
        print(seq_len, "seq_len")
        print(start_position, max([start_position-seq_len,0]), start_position-seq_len+dataset.window_size)
        s = dataset.padded_data[index, max([start_position-seq_len,0]):start_position-seq_len+dataset.window_size]
        if start_position == (seq_len+1):
            target = dataset.END_IDX
        else:
            target = dataset.padded_data[index, start_position-seq_len+dataset.window_size]

        # print(target)

        return {'input': torch.Tensor(s).long(),
                'target': torch.Tensor([target]).long()}

    def __len__(self):
        return len(self.data)

    def reset(self):
        self.start_positions = {}

    #get words id
    def get_sentence_idx(self, index):
        dataset = self


class TrajectorySelfAttentionDataset(Dataset):
    #Init dataset
    def __init__(self, data, window_size, seq_len, n_bins, dataset_name="dataset", random_mask=False):
        dataset = self
        
        dataset.data = data
        nan_place = np.isnan(data)
        dataset.seq_lens = np.sum(nan_place==False, axis=1)

        dataset.n_locations = (n_bins+2)**2
        dataset.n_locations_plus_end = dataset.n_locations + 1
        dataset.vocab = list(range(dataset.n_locations)) + ['<end>', '<start>', '<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e:i for i, e in enumerate(dataset.vocab)} 
        dataset.seq_len = seq_len
        dataset.random_mask = random_mask
        dataset.dataset_name = dataset_name
        dataset.window_size = window_size
        dataset.n_vocab = len(dataset.vocab)
        
        print("random_mask:", dataset.random_mask)
        #special tags
        dataset.START_IDX = dataset.vocab['<start>']
        dataset.IGNORE_IDX = dataset.vocab['<ignore>'] #replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>'] #replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>'] #replacement tag for the masked word prediction task
        dataset.END_IDX = dataset.vocab['<end>']

        # The data includes nan values to pad the sequences to the same length.
        # We replace the nan values with the special tag IGNORE_IDX and add the special tag END_IDX to the end of each sequence
        dataset.data[nan_place] = dataset.END_IDX
        dataset.data = np.concatenate([dataset.data, np.array([[dataset.END_IDX]]*len(dataset.data))], axis=1).astype(int)
        IGNORE_IDX_place = np.concatenate([np.array([[False]]*len(dataset.data)),nan_place], axis=1)
        dataset.data[IGNORE_IDX_place] = dataset.IGNORE_IDX

        # insert dataset.START_IDX to the first position
        dataset.data = np.concatenate([np.array([[dataset.START_IDX]]*len(dataset.data)),dataset.data], axis=1).astype(int)

    def __str__(self):
        return self.dataset_name
        
    #fetch data
    # def __getitem__(self, index, p_random_mask=0.15):
    def __getitem__(self, index):
        dataset = self

        # nan_place = np.isnan(dataset.data[index])
        # seq_len = np.sum(nan_place==False)
        seq_len = dataset.seq_lens[index]
        seq_len_include_start_end = seq_len+2


        # if dataset.window_size < seq_len_include_start_end:
        start_position = np.random.choice(range(0,max([seq_len_include_start_end-dataset.window_size-1,1])))
        # else:
        #     start_position = 0

        s = dataset.data[index, start_position:start_position+dataset.window_size+1]
        target = dataset.data[index, start_position+1:start_position+dataset.window_size+2]
        
        return {'input': torch.Tensor(s).long(),
                'target': torch.Tensor(target).long()}

    def __len__(self):
        return len(self.data)

    #get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.data[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s] 
        return s

    def reset(self):
        pass
    
    
    
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