import pandas as pd
import json
import numpy as np
import bisect
import random
from sklearn.preprocessing import normalize
import pathlib
from dataset import TrajectoryDataset
import torch


# def generate_samples(model, batch_size, seq_len, generated_num, output_file, real_start=None):
#     samples = []
#     for i in range(int(generated_num / batch_size)):
#         if real_start is not None:
#             sample = model.sample(batch_size, start_time=1, data=real_start[i*batch_size:(i+1)*batch_size]).cpu().data.numpy().tolist()
#         else:
#             sample = model.sample(batch_size).cpu().data.numpy().tolist()
#         samples.extend(sample)
        
#     df = pd.DataFrame(samples)
#     df.to_csv(output_file, header=None, index=None)


def load_dataset(dataset, data_name, window_size, mask_start=1, random_mask=False):
    
    data_path = get_datadir() / dataset / data_name
    data = pd.read_csv(data_path / "training_data.csv", header=None).values
    
    if window_size == 0:
        window_size = len(data[0])-1

    print(f"load data from {data_path / 'training_data.csv'}")
    
    print(f'creating dataset..., mask_start {mask_start}')
    
    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    
    dataset = TrajectoryDataset(data, window_size, len(data[0]), n_bins, f"{dataset}/{data_name}", random_mask, mask_start)
    return dataset
                 
                 
def get_datadir():
    with open(f"config.json", "r") as f:
        config = json.load(f)
    return pathlib.Path(config["data_dir"])



def get_gps(dataset):
    df = pd.read_csv(get_datadir() / f"{dataset}/gps.csv", header=None)
    return df.values[:,1], df.values[:,0]

def get_maxdistance(dataset):
    X, Y = get_gps(dataset)
    dx = X[0] - X[-1]
    dy = Y[0] - Y[-1]
    return np.sqrt(dx**2 + dy**2)

def make_gps(lat_range, lon_range, n_bins):
    
    x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+2)
    y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+2)
    
    def state_to_latlon(state):
        x_state = int(state % (n_bins+2))
        y_state = int(state / (n_bins+2))
        return y_axis[y_state], x_axis[x_state]
    
    return pd.DataFrame([state_to_latlon(i) for i in range((n_bins+2)**2)])


def load_M1(dataset):
    return normalize(np.load(get_datadir() / f'{dataset}/M1.npy'))

def load_M2(dataset):
    return normalize(np.load(get_datadir() / f'{dataset}/M2.npy'))

def construct_M1(training_data, max_locs):
    reg1 = np.zeros([max_locs,max_locs])
    for line in training_data:
        for j in range(len(line)-1):
            if (line[j] >= max_locs) or (line[j+1] >= max_locs):
#                 print("WARNING: outside location found")
                continue
            reg1[line[j],line[j+1]] +=1
    return reg1

def construct_M2(train_data, max_locs, gps):
    xs = gps[0]
    ys = gps[1]

    reg2 = []
    for (x,y) in zip(xs,ys):
        reg2.append(np.sqrt((ys - y)**2 + (xs - x)**2))
    reg2 = np.array(reg2)  
    return reg2



def load_latlon_range(name):
    with open(f"{name}.json", "r") as f:
        configs = json.load(f)
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    return lat_range, lon_range
    
def latlon_to_state(lat, lon, lat_range, lon_range, n_bins):
    x_axis = np.linspace(lon_range[0], lon_range[1], n_bins+1)
    y_axis = np.linspace(lat_range[0], lat_range[1], n_bins+1)
    
    x_state = bisect.bisect_left(x_axis, lon)
    y_state = bisect.bisect_left(y_axis, lat)
    # if x_state == n_bins+2:
    #     x_state = n_bins
    # if y_state == n_bins+2:
    #     y_state = n_bins
    return y_state*(n_bins+2) + x_state

def make_hist_2d(counts, n_bins):
    hist2d = [[0 for i in range(n_bins+2)] for j in range(n_bins+2)]
    for state in range((n_bins+2)**2):
        x,y = state_to_xy(state, n_bins)
        hist2d[x][y] = counts[state]
    return np.array(hist2d)

def state_to_xy(state, n_bins):
    n_x = n_bins+2
    n_y = n_bins+2

    x = (state) % n_x
    y = int((state) / n_y)
    return x, y

def split_train_test(df, seed, split_ratio=0.5):
    random.seed(seed)
    n_records = len(df.index)
    choiced_indice = random.sample(range(n_records), int(n_records*split_ratio))
    removed_indice = [i for i in range(n_records) if i not in choiced_indice]
    training_df = df.loc[choiced_indice]
    test_df = df.loc[removed_indice]
    return training_df, test_df


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss