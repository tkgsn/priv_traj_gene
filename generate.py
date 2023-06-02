import argparse
import random
import numpy as np
import sys
import torch
import pathlib
import tqdm
import datetime
from torch import nn, optim
import json
import pandas as pd
import glob
from sklearn.metrics import roc_auc_score
from meta import meta_learning, pre_training

from my_utils import load_latlon_range, make_gps, load_M1, load_M2, get_datadir, load_dataset
from dataset import RealFakeDataset, TrajectoryDataset, TrajectorySelfAttentionDataset, padded_collate_without_end, make_padded_collate, int_to_float_of_minute, convert_time_traj_to_time_traj_float, TrajectoryDataset_with_Time, traj_to_format
from models import Discriminator, Transformer, SelfAttentionTransformer, TimeGRUNet, GRUNet, DPGRUNet, make_sample, make_sample_with_time, make_sample_with_time_and_traj_type, TimeTrajTypeGRUNet, MetaTimeTrajTypeGRUNet
from evaluation import evaluation
from rollout import Rollout
from loss import GANLoss
from grid import Grid
from data_processing import save_latlon_with_nan_padding, save_state_with_nan_padding
import torch.nn.functional as F

from opacus import PrivacyEngine
from pytorchtools import EarlyStopping

import random
from argparse import Namespace
import numpy as np
import torch
from my_utils import get_datadir, load_dataset
import json
import glob
from run_meta_learning import compute_traj_type_distribution

args = Namespace()
args.seed = 1
args.dataset = "peopleflow"
args.data_name = "peopleflow_98_time20_loc500"
args.save_name = "peopleflow_98_time20_loc500_100000_dp_nlearn1_noise0.5"
args.end_epoch = 461
args.end_loss = 7.081544
args.max_size = 100000

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms = True
torch.backends.cudnn.deterministic = True

data_dir = get_datadir()

save_path = data_dir / "results" / args.dataset / args.save_name
data_path = data_dir / f"{args.dataset}" / f"{args.data_name}"

save_path.mkdir(exist_ok=True, parents=True)

with open(data_path / "params.json", "r") as f:
    param = json.load(f)
n_bins = param["n_bins"]
lat_range = param["lat_range"]
lon_range = param["lon_range"]

if (data_path / "training_data_0000.csv").exists():
    # load data from data_path training_data_*.csv
    data_dirs = glob.glob(str(data_path / "training_data_0*.csv"))
    # sort data_dirs
    data_dirs = sorted(data_dirs)
    time_data_dirs = [str(data_path / f"training_data_time_{i:04d}.csv") for i in range(len(data_dirs))]

    assert len(data_dirs) == len(time_data_dirs)

    real_time_traj = load_dataset(time_data_dirs)
    trajectories = load_dataset(data_dirs)

    # a = [len(traj) for traj in trajectories]
    # b = [len(traj) for traj in real_time_traj]
    # print("num bug", sum([a[i]+1!=b[i] for i in range(len(a))]))

    # remove trajs including None
    # remove time trajs whose index is None in trajectories
    real_time_traj = [real_time_traj[i] for i in range(len(trajectories)) if "None" not in trajectories[i]]
    trajectories = [trajectory for trajectory in trajectories if "None" not in trajectory]
    # real_time_traj = [[(float(v.split("_")[0]), float(v.split("_")[1])) for v in trajectory if v != 'nan'] for trajectory in real_time_traj]
    real_time_traj = [[float(v) for v in trajectory if v != 'nan'] for trajectory in real_time_traj]
    trajectories = [[int(float(v)) for v in trajectory if v != 'nan'] for trajectory in trajectories]

    real_time_traj = [real_time_traj[i] for i in range(len(trajectories)) if len(trajectories[i]) > 0]
    trajectories = [trajectory for trajectory in trajectories if len(trajectory) > 0]

    print(f"len of trajectories: {len(trajectories)}")
    print(f"len of real_time_traj: {len(real_time_traj)}")
    assert len(trajectories) == len(real_time_traj)

    if args.max_size == 0:
        args.max_size = len(trajectories)
    # shuffle trajectories and real_time_traj with the same order without using numpy
    p = np.random.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in p][:args.max_size]
    real_time_traj = [real_time_traj[i] for i in p][:args.max_size]

    # a = [len(traj) for traj in trajectories]
    # b = [len(traj) for traj in real_time_traj]
    # print("num bug", sum([a[i]+1!=b[i] for i in range(len(a))]))


    # print(real_time_traj)
    # print(trajectories)

    label_count, format_to_label, label_to_format = compute_traj_type_distribution(trajectories)

    print(f"len of cut trajectories: {len(trajectories)}")
    print(f"len of cut real_time_traj: {len(real_time_traj)}")
    assert len(trajectories) == len(real_time_traj)


dataset = TrajectoryDataset_with_Time(trajectories, real_time_traj, n_bins, format_to_label)

args.hidden_dim = 128
args.embed_dim = 256
args.n_layers = 1
args.cuda_number = 0

input_dim = dataset.n_locations+2
output_dim = dataset.n_locations
hidden_dim = args.hidden_dim
embed_size = args.embed_dim
n_layers = args.n_layers
traj_type_dim = len(label_count)
print("input_dim", input_dim, "traj_type_dim", traj_type_dim, "hidden_dim", hidden_dim, "output_dim", output_dim, "n_layers", n_layers, "embed_size", embed_size)
# generator = MetaTimeTrajTypeGRUNet(input_dim, traj_type_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
generator = TimeTrajTypeGRUNet(input_dim, traj_type_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)

privacy_engine = PrivacyEngine(accountant='rdp')
poisson_sampling = True
args.lr = 1e-4
args.batch_size = 50
kwargs = {'num_workers':0, 'shuffle':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':make_padded_collate(dataset.IGNORE_IDX, dataset.START_IDX, format_to_label)}
optim_kwargs = {'lr':args.lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

args.dp_noise_multiplier = 0.1
args.clipping_bound = 1
args.loss_reduction = "mean"

generator, optimizer, data_loader = privacy_engine.make_private(
    module=generator,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=args.dp_noise_multiplier,
    max_grad_norm=args.clipping_bound,
    loss_reduction=args.loss_reduction,
    poisson_sampling=poisson_sampling
)


if not (save_path / "params.json").exists():
    print("save params", save_path / "params.json")
    with open(save_path / "params.json", "w") as f:
        json.dump(vars(args), f)

generator.load_state_dict(torch.load(save_path / "checkpoint.pt"))
generator.eval()

args.n_generated = len(dataset)
dataset_labels = torch.tensor([format_to_label[traj_to_format(trajectory)] for trajectory in trajectories])
dataset_labels = torch.Tensor(np.random.choice(dataset_labels, args.n_generated, replace=True)).long()
print(dataset_labels)
# print(dataset_labels)
real_start = False
samples, time_samples = make_sample_with_time_and_traj_type(args.batch_size, generator._module, dataset_labels, dataset, label_to_format, real_start=real_start)

generated_data_path = save_path/f"gene.csv"
generated_time_data_path = save_path/f"gene_time.csv"

max_time = 24*60

for time_sample in time_samples:
    for i in range(len(time_sample)):
        time_sample[i] = int(time_sample[i]*max_time)

save_state_with_nan_padding(generated_data_path, samples)
save_state_with_nan_padding(generated_time_data_path, time_samples)