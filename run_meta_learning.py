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

    
def locations_to_format_label(trajs):
    trajs = trajs.tolist()
    # remove ignore_idx
    trajs = [[location for location in traj if location != trajectory_dataset.IGNORE_IDX] for traj in trajs]
    # convert to format
    formats = [traj_to_format(traj) for traj in trajs]
    # convert to label
    labels = [format_to_label[format] for format in formats]
    return labels

def make_format_to_label(traj_list):
    format_to_label = {}
    for trajectory in traj_list:
        traj_type = traj_to_format(trajectory)
        if traj_type not in format_to_label:
            format_to_label[traj_type] = len(format_to_label)
    return format_to_label

def make_label_to_format(format_to_label):
    label_to_format = {}
    for format in format_to_label:
        label = format_to_label[format]
        label_to_format[label] = format
    return label_to_format
    
def make_fotmat_time_to_label(time_traj_list):
    # make dictionary that maps a format to a label
    format_time_to_label = {}
    for time_traj in time_traj_list:
        format_time = time_traj_to_format(time_traj)
        if format_time not in format_time_to_label:
            format_time_to_label[format_time] = len(format_time_to_label)
    return format_time_to_label


def locations_to_format_label(trajs):
    # trajs = trajs.tolist()
    # remove ignore_idx
    trajs = [[location for location in traj if location != dataset.IGNORE_IDX] for traj in trajs]
    # convert to format
    formats = [traj_to_format(traj) for traj in trajs]
    # convert to label
    labels = [format_to_label[format] for format in formats]
    return labels

def compute_traj_type_distribution(real_traj):
    # make dictionary that maps a format to a label
    format_to_label = make_format_to_label(real_traj)
    # label_to_format
    label_to_format = make_label_to_format(format_to_label)

    # make a list of labels
    label_list = [format_to_label[traj_to_format(trajectory)] for trajectory in real_traj]

    # count the number of trajectories for each label
    label_count = [0] * len(format_to_label)
    for label in label_list:
        label_count[label] += 1
    
    # normalize
    label_count = [count / sum(label_count) for count in label_count]
    return label_count, format_to_label, label_to_format

def load_and_process_dataset(dataset, data_name):

    data_path = pathlib.Path(get_datadir()) / dataset / data_name
    data_dirs = sorted(glob.glob(str(data_path / "training_data_0*.csv")))
    time_data_dirs = [str(data_path / f"training_data_time_{i:04d}.csv") for i in range(len(data_dirs))]

    assert len(data_dirs) == len(time_data_dirs)
    for data_dir, time_data_dir in zip(data_dirs, time_data_dirs):
        print(data_dir, time_data_dir)

    real_traj = load_dataset(data_dirs)
    real_time_traj = load_dataset(time_data_dirs)

    real_time_traj = [real_time_traj[i] for i in range(len(real_traj)) if "None" not in real_traj[i]]
    real_traj = [trajectory for trajectory in real_traj if "None" not in trajectory]
    real_time_traj = [[(int(v.split("_")[0]), int(v.split("_")[1])) for v in trajectory if v != 'nan'] for trajectory in real_time_traj]
    real_traj = [[int(float(v)) for v in trajectory if v != 'nan'] for trajectory in real_traj]

    return real_traj, real_time_traj

# define loss function on time prediction by MSE
def loss_time_fn(pred, target, ignore_value):
    # if target is ignore_value, we don't want to calculate the loss
    mask = (target != ignore_value).float()
    pred = pred * mask
    target = target * mask
    loss = nn.functional.mse_loss(pred, target)
    return loss
    
def loss_kl(output_location, global_distribution):

    # loss = 0
    # p_r1 = global_distributions[1]
    # for output_distribution in output_locations:
    #     output_distribution = output_distribution[1]
    #     loss += F.kl_div(output_distribution.log(), p_r1, None, None, 'sum')

    # p_r1 = global_distributions[1]
    # output_distribution = torch.zeros(p_r1.shape).cuda(p_r1.device)
    # for each_output_distribution in output_locations:
        # output_distribution += each_output_distribution[1]/len(output_locations)
    # loss = F.kl_div(output_distribution.log(), p_r1, None, None, 'sum')
    loss = F.kl_div(output_location.log(), global_distribution, None, None, 'sum')

    return loss

def compute_global_distribution(trajectories, real_time_traj, time, n_locations):
    def location_at_time(trajectory, time_traj, t):
        # find the index where time range includes t
        for i in range(len(trajectory)):
            if int(time_traj[i]) <= t and t < int(time_traj[i+1]):
                return int(trajectory[i])
        print(trajectory, time_traj, t)
            
    locations = []
    count = 0
    for trajectory, time_traj in zip(trajectories, real_time_traj):
        if 1+len(trajectory) != len(time_traj):
            # print("BUG, NEED TO BE FIXED", trajectory, time_traj)
            count += 1
        else:
            locations.append(location_at_time(trajectory, time_traj, time))
    # print("BUG COUNT", count)

    # print(time, locations)
    # count each location and conver to probability
    location_count = {i:0 for i in range(n_locations)}
    for location in locations:
        location_count[location] += 1
    location_prob = {location: count / len(locations) for location, count in location_count.items()}
    return list(location_prob.values())

def laplace_mechanism(x, epsilon):
    if epsilon == 0:
        print("laplace is not used")
        return x
    return x + np.random.laplace(0, 1/epsilon, len(x))


def compute_noisy_global_distribution(trajectories, real_time_traj, time, n_locations, epsilon):
    def location_at_time(trajectory, time_traj, t):
        # find the index where time range includes t
        for i in range(len(trajectory)):
            if int(time_traj[i]) <= t and t < int(time_traj[i+1]):
                return int(trajectory[i])
        print(trajectory, time_traj, t)
            
    locations = []
    count = 0
    for trajectory, time_traj in zip(trajectories, real_time_traj):
        if 1+len(trajectory) != len(time_traj):
            # print("BUG, NEED TO BE FIXED", trajectory, time_traj)
            count += 1
        else:
            locations.append(location_at_time(trajectory, time_traj, time))
    # print("BUG COUNT", count)

    # print(time, locations)
    # count each location and conver to probability
    location_count = {i:0 for i in range(n_locations)}
    for location in locations:
        location_count[location] += 1
    # location_prob = {location: count / len(locations) for location, count in location_count.items()}
    location_prob = {location: count for location, count in location_count.items()}
    noised_distribution = laplace_mechanism(np.array(list(location_prob.values())), epsilon)
    # the minus value is clipped to 0
    noised_distribution = np.clip(noised_distribution, 0, None)
    # normalize the distribution
    noised_distribution = noised_distribution / sum(noised_distribution)

    return noised_distribution


def train_with_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels, is_dp, n_learn=0):
    generator.train()

    # concat input_locations and target_locations
    # test = torch.cat([input_locations, target_locations], dim=1)
    # print(test)
    # print(labels)

    # print(target_locations)

    output_locations, output_times = generator([input_locations, input_times], labels)
    output_locations_v = output_locations.view(-1,output_locations.shape[-1])
    output_times_v = output_times.view(-1,output_times.shape[-1])
    # print(output_v.shape)
    # test = torch.exp(output_locations_v)
    # print(test[0][5])
    # print(test[0][7])

    # randomly cover a target location so that the number of learned location is n_learn
    if n_learn != 0:
        for target in target_locations:
            while sum(target != dataset.IGNORE_IDX) > n_learn:
                # find locations that are not ignore_idx
                not_ignore_idx = [i for i in range(len(target)) if target[i] != dataset.IGNORE_IDX]
                # randomly choose one location
                idx = random.choice(not_ignore_idx)
                # cover the location with ignore_idx
                target[idx] = dataset.IGNORE_IDX
        # print(target_locations)
    
    # print(target_locations)

    loss_location = loss_model(output_locations_v, target_locations.view(-1))
    loss_time = loss_time_fn(output_times_v, target_times, ignore_value=dataset.IGNORE_IDX)
    # print(loss_time.item(), loss_location.item())
    loss_time = 0
    loss = loss_location + loss_time
    loss.backward()

    # WARNING!!!!!
    # I think it's ok, but it's possible that this includes bugs because of the gradient accumulation
    # print(generator.embeddings.weight.grad_sample.shape)
    # print(optimizer.grad_samples.shape)

    # if is_dp:
    #     # count the number of location in each trajectory
    #     counts = []
    #     for target_location in target_locations:
    #         counts.append(sum(target_location != dataset.IGNORE_IDX))


    #     # divide gradients by training length (except for ignore_value)
    #     for param in generator.parameters():
    #         # print(len(counts))
    #         for i in range(len(counts)):
    #             param.grad_sample[i] /= counts[i]

    optimizer.step()
    optimizer.zero_grad()

    return loss.item()

def train_with_time_test_bias(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels):
    generator.train()

    seq_len = input_locations.shape[1]
    output_locations, output_times = generator([input_locations, input_times], labels)
    output_locations_v = output_locations.view(-1,output_locations.shape[-1])
    output_times_v = output_times.view(-1,output_times.shape[-1])
    # print(output_v.shape)
    loss_location = loss_model(output_locations_v, target_locations.view(-1))
    loss_time = loss_time_fn(output_times_v, target_times, ignore_value=dataset.IGNORE_IDX)
    loss = loss_location + loss_time
    loss.backward()
    # get per sample gradient
    # print(generator.embeddings.weight.grad_sample.shape)

    # WARNING!!!!!
    # I think it's ok, but it's possible that this includes bugs because of the gradient accumulation
    # print(generator.embeddings.weight.grad_sample.shape)
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()



def train(generator, optimizer, loss_model, input, target):
    generator.train()

    output = generator(input)
    output_v = output.view(-1,output.shape[-1])
    print(output_v.shape)
    loss = loss_model(output_v, target)
    loss.backward()

    # WARNING!!!!!
    # I think it's ok, but it's possible that this includes bugs because of the gradient accumulation

    # print(generator.embeddings.weight.grad)
    optimizer.step()
    # grad_norm = round(generator.embeddings.weight.grad.abs().sum().item(),3)
    optimizer.zero_grad()

    return loss.item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number',  default=0, type=int)
    parser.add_argument('--print_epoch',  default=10, type=int)
    parser.add_argument('--n_generated',  default=0, type=int)
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--data_name', default='test_1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--n_learn', default=0, type=int)
    parser.add_argument('--meta_batch_size', default=1000, type=int)
    parser.add_argument('--meta_n_iter', default=10, type=int)
    parser.add_argument('--meta_interval', default=100, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--window_size', default=0, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--save_name', default="", type=str)
    parser.add_argument('--loss_reduction', default="mean", type=str)
    parser.add_argument('--accountant_mode', default="prv", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--meta_lr', default=1e-4, type=float)
    parser.add_argument('--dp_noise_multiplier', default=1, type=float)
    parser.add_argument('--clipping_bound', default=1.0, type=float)
    parser.add_argument('--epsilon', default=4.0, type=float)
    parser.add_argument('--n_split', default=4, type=int)
    parser.add_argument('--global_epsilon_ratio', default=0.5, type=float)
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--is_dp', action='store_true')
    parser.add_argument('--is_traj_type', action='store_true')
    parser.add_argument('--is_time', action='store_true')
    parser.add_argument('--is_self_attention', action='store_true')
    parser.add_argument('--is_evaluation', action='store_true')
    parser.add_argument('--without_end', action='store_true')
    parser.add_argument('--only_first_markov', action='store_true')
    parser.add_argument('--is_test_bias', action='store_true')
    parser.add_argument('--is_meta', action='store_true')
    parser.add_argument('--is_pre_training', action='store_true')
    parser.add_argument('--is_kl_loss', action='store_true')
    parser.add_argument('--max_size', default=0, type=int)
    parser.add_argument('--patience', default=5, type=int)


    args = parser.parse_args()
    
    if args.save_name == "":
        args.save_name = args.data_name
    
        
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

    else:
        print("make training data")
        make_training_data(data_path, n_bins)

    # original_data_path = data_path / "original_data.csv"

    # compute max seq len
    max_seq_len = max([len(trajectory) for trajectory in trajectories])
    print(f"max seq len: {max_seq_len}")
    
    # if args.window_size == 0:
    #     args.window_size = max_seq_len
    #     print(f"window size is set as {args.window_size}")

    # make fake format_to_label
    fake_format_to_label = False
    if fake_format_to_label:
        print("WARNING: fake format_to_label is used")
        for key in format_to_label.keys():
            format_to_label[key] = 0


    if args.is_time:
        dataset = TrajectoryDataset_with_Time(trajectories, real_time_traj, n_bins, format_to_label)
        # if args.is_meta:
        #     meta_dataset = TrajectoryDataset_with_Time(meta_traj, meta_time_traj, n_bins, format_to_label)
    elif args.is_self_attention:
        dataset = TrajectorySelfAttentionDataset(data, args.window_size, max_seq_len, n_bins, f"{args.dataset}/{args.data_name}", random_mask=False)
    else:
        # dataset = TrajectoryDataset(trajectories, args.window_size, max_seq_len, n_bins, f"{args.dataset}/{args.data_name}", random_mask=False)
        dataset = TrajectoryDataset(trajectories, n_bins)
    n_vocabs = len(dataset.vocab)

    if args.n_generated == 0:
        args.n_generated = len(dataset)
        print("generating " + str(args.n_generated) + " samples")
    
    if args.batch_size == 0:
        args.batch_size = int(np.sqrt(len(dataset)))
        print("batch size is set as " + str(args.batch_size))

    # ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    # grid = Grid(ranges)

    # generator = TransGenerator_DP(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)
    # generator = Transformer(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)

    dataset_labels = torch.tensor([format_to_label[traj_to_format(trajectory)] for trajectory in trajectories])

    if args.gru:
        input_dim = dataset.n_locations+2
        output_dim = dataset.n_locations
        hidden_dim = args.hidden_dim
        embed_size = args.embed_dim
        n_layers = args.n_layers
        traj_type_dim = len(label_count)
        if args.is_meta or args.is_pre_training:
            print("input_dim", input_dim, "traj_type_dim", traj_type_dim, "hidden_dim", hidden_dim, "output_dim", output_dim, "n_layers", n_layers, "embed_size", embed_size)
            generator = MetaTimeTrajTypeGRUNet(input_dim, traj_type_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
            temp_generator = MetaTimeTrajTypeGRUNet(input_dim, traj_type_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
        elif args.is_traj_type:
            generator = TimeTrajTypeGRUNet(input_dim, traj_type_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
        elif args.is_time:
            generator = TimeGRUNet(input_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
        else:
            generator = DPGRUNet(input_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
        # else:
        #     generator = GRUNet(input_dim, hidden_dim, output_dim, n_layers, embed_size, drop_prob=0).cuda(args.cuda_number)
    else:
        embed_size = args.embed_dim
        inner_ff_size = embed_size*4
        n_heads = 1
        n_code = 1

        if args.is_self_attention:
            # the generator has additional vocabraries start, end, and ignore
            generator = SelfAttentionTransformer(n_code, n_heads, embed_size, inner_ff_size, n_vocabs, n_vocabs, max_seq_len+1, 0.1).cuda(args.cuda_number)
        else:
            generator = Transformer(n_code, n_heads, embed_size, inner_ff_size, n_vocabs, dataset.n_locations+1, max_seq_len+1, dataset.CLS_IDX, 0.1).cuda(args.cuda_number)
        # generator.real = False


    cuda_number = next(generator.parameters()).device.index
    
    if args.is_test_bias:
        print("TESTING BIAS")

    if args.is_time:
        print("INCLUDE TIME")
        kwargs = {'num_workers':0, 'shuffle':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':make_padded_collate(dataset.IGNORE_IDX, dataset.START_IDX, format_to_label)}
    elif args.only_first_markov:
        print("only first markov")
        kwargs = {'num_workers':0, 'shuffle':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':padded_collate_without_end_only_first_markov}
    elif args.without_end:
        kwargs = {'num_workers':0, 'shuffle':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':padded_collate_without_end}
    else:    
        kwargs = {'num_workers':0, 'shuffle':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':padded_collate}
    # kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    # if args.is_meta or args.is_pre_training:
    #     meta_kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.meta_batch_size, 'collate_fn':make_padded_collate(dataset.IGNORE_IDX, dataset.START_IDX, format_to_label)}
    #     meta_data_loader = torch.utils.data.DataLoader(meta_dataset, **meta_kwargs)

    optim_kwargs = {'lr':args.lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
    optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
    # loss_model = nn.NLLLoss(ignore_index=dataset.START_IDX)
    print("IGNORE", dataset.IGNORE_IDX)
    loss_model = nn.NLLLoss(ignore_index=dataset.IGNORE_IDX)
    
    print_epoch = args.print_epoch
    
    max_time = 24*60-1
    # time_ranges := [(0, 1/n_split), (1/n_split, 2/n_split), ..., ((n_split-1)/n_split, 1)]
    n_split = args.n_split
    time_ranges = [(i/n_split, (i+1)/n_split) for i in range(n_split)]
    epsilon_for_global = args.epsilon * args.global_epsilon_ratio
    epsilon_for_sgd = args.epsilon * (1-args.global_epsilon_ratio)
    global_distributions = torch.tensor([compute_noisy_global_distribution(trajectories, real_time_traj, time, dataset.n_locations, epsilon_for_global/n_split) for time in [i[0]*max_time for i in time_ranges]]).to(cuda_number)
    print("GLOBAL DISTRIBUTIONS", global_distributions, f"split by {n_split}", "epsilon", epsilon_for_global)
    
    dataset.data_loader = data_loader
    if args.is_dp:
        print("IS DP")
        privacy_engine = PrivacyEngine(accountant=args.accountant_mode)
        poisson_sampling = True
        generator, optimizer, data_loader = privacy_engine.make_private(
            module=generator,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=args.dp_noise_multiplier,
            max_grad_norm=args.clipping_bound,
            loss_reduction=args.loss_reduction,
            poisson_sampling=poisson_sampling
        )
        delta = 1e-5
        epsilons = []

    # generator = meta_learning(data_loader, temp_generator, generator, loss_model, loss_time_fn, dataset.IGNORE_IDX, args.lr, loss_kl, global_distributions)

    if args.is_dp:
        eval_generator = generator._module
    else:
        eval_generator = generator

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path / "checkpoint.pt")
    print("USE EARLY STOPPING", f"patience: {args.patience}", f"save path: {save_path / 'checkpoint.pt'}")

    with open(save_path / "params.json", "w") as f:
        json.dump(vars(args), f)

    print("save param to ", save_path / "params.json")
    print(vars(args))
    
    if args.epsilon == 0:
        epsilon_for_sgd = float("inf")

    for epoch in tqdm.tqdm(range(args.n_epochs)):


        dataset.reset()
        loss = 0
        counter = 0
        for i, batch in enumerate(data_loader):

            if ((epoch * len(data_loader) + i) % args.meta_interval == 0) and (args.is_meta or args.is_pre_training):
                generator.eval()
                dataset_labels = torch.Tensor(np.random.choice(dataset_labels, args.meta_batch_size * args.meta_n_iter, replace=True)).long()
                real_start = False
                meta_traj, meta_time_traj =  make_sample_with_time_and_traj_type(args.batch_size, eval_generator, dataset_labels, dataset, label_to_format, real_start=real_start)
                # meta_traj[meta_traj == dataset.END_IDX] = dataset.IGNORE_IDX
                for time_sample in meta_time_traj:
                    for j in range(len(time_sample)):
                        time_sample[j] = int(time_sample[j]*max_time)
                generator.train()
                meta_dataset = TrajectoryDataset_with_Time(meta_traj, meta_time_traj, n_bins, format_to_label)
                print("dataset size for meta learning", len(meta_dataset))
                meta_kwargs = {'num_workers':0, 'shuffle':True, 'pin_memory':True, 'batch_size':args.meta_batch_size, 'collate_fn':make_padded_collate(dataset.IGNORE_IDX, dataset.START_IDX, format_to_label)}
                meta_data_loader = torch.utils.data.DataLoader(meta_dataset, **meta_kwargs)
                # noise_scale = args.dp_noise_multiplier * args.clipping_bound / args.batch_size
                noise_scale = 0
                if args.is_pre_training:
                    print("iter", epoch * len(data_loader) + i, "pre training")
                    generator = pre_training(meta_data_loader, temp_generator, generator, args.meta_lr, global_distributions, time_ranges, loss_kl, dataset.IGNORE_IDX, args.meta_n_iter)
                elif args.is_meta:
                    print("iter", epoch * len(data_loader) + i, "meta learning")
                    generator = meta_learning(meta_data_loader, temp_generator, generator, loss_model, loss_time_fn, dataset.IGNORE_IDX, args.meta_lr, loss_kl, global_distributions, noise_scale, time_ranges, args.meta_n_iter)


            if len(batch["input"]) == 0:
                continue
            counter += 1
            # print(i)
            # input = batch['input'].reshape(len(batch['input']),-1).cuda(cuda_number, non_blocking=True)
            # target = batch['target'].cuda(cuda_number, non_blocking=True).reshape(-1)

            # indices = batch["index"]
            # raw_data = [trajectories[index] for index in indices if len(trajectories[index]) > 1]

            input_locations = batch["input"].cuda(cuda_number, non_blocking=True)

            batch_size = input_locations.shape[0]
            seq_len = input_locations.shape[1]

            target_locations = batch["target"].cuda(cuda_number, non_blocking=True)
            labels = batch["label"].cuda(cuda_number, non_blocking=True)
            # print(labels)
            # print([label_to_format[label.item()] for label in labels])

            # labels = torch.tensor(locations_to_format_label(raw_data)).cuda(cuda_number, non_blocking=True)
            # labels = labels.fill_(0)

            
            if args.is_time:
                input_times = batch["time"].reshape(batch_size, seq_len, 1).cuda(cuda_number, non_blocking=True)
                target_times = batch["time_target"].cuda(cuda_number, non_blocking=True)
                if args.is_test_bias:
                    # print("TESTING BIAS")
                    loss += train_with_time_test_bias(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels)
                else:
                    # print("skip")
                    loss += train_with_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels, args.is_dp, args.n_learn)
            else:
                loss += train(generator, optimizer, loss_model, input, target)
            
            # break
        early_stopping(loss / counter, generator)

        if args.is_dp:
            epsilon = privacy_engine.get_epsilon(delta)
            epsilons.append(epsilon)
            print(f'epsilon{epsilon}/{epsilon_for_sgd}, delta{delta}, loss {loss/(i+1)}')
        else:
            epsilon = 0

        if (epoch != 0 and epoch % print_epoch == 0) or early_stopping.early_stop or epsilon > epsilon_for_sgd:
            
            if early_stopping.early_stop:
                generator.load_state_dict(torch.load(early_stopping.path))
                generator.eval()
                print('epoch:', early_stopping.epoch, ' | loss', early_stopping.best_score)
            else:
                print('epoch:', epoch, ' | loss', loss/(i+1))

            if args.is_evaluation:
                
                generated_data_path = save_path / f"gene.csv"
                generated_time_data_path = save_path / f"gene_time.csv"
                # wo_generated_data_path = save_path / f"without_real_gene.csv"
                # real_start = generator.make_initial_data(len(dataset))
                # real_start[:,generator.window_size] = torch.tensor(dataset.data[:, 0])
                # print(f"generate to {generated_data_path}")
                # generate_samples(generator, batch_size, dataset.seq_len, len(dataset), generated_data_path, real_start)
                # print(dataset_labels)
                # n_sample = 
                if args.is_traj_type:
                    # print(dataset_labels)
                    # dataset_labels = torch.Tensor([0]*args.n_generated).long()
                    dataset_labels = torch.Tensor(np.random.choice(dataset_labels, args.n_generated, replace=True)).long()
                    print(dataset_labels)
                    # print(dataset_labels)
                    real_start = False
                    samples, time_samples = make_sample_with_time_and_traj_type(args.batch_size, generator, dataset_labels, dataset, label_to_format, real_start=real_start)
                elif args.is_time:
                    samples, time_samples = make_sample_with_time(args.batch_size, eval_generator, args.n_generated, dataset)
                else:
                    samples = make_sample(args.batch_size, eval_generator, args.n_generated, dataset)

                for time_sample in time_samples:
                    for i in range(len(time_sample)):
                        time_sample[i] = int(time_sample[i]*max_time)

                save_state_with_nan_padding(generated_data_path, samples)
                save_state_with_nan_padding(generated_time_data_path, time_samples)
                # print(samples)
                # samples_wo_start = make_sample(args.batch_size, eval_generator, n_sample, dataset, real_start=False)
                # pd.DataFrame(samples).to_csv(generated_data_path, header=None, index=None)

                # print("a", samples)
                # if is_dp:
                #     # samples = make_sample(batch_size, generator._module, n_sample, dataset)
                #     samples_wo_start = make_sample(batch_size, generator._module, n_sample, dataset, real_start=False)
                # else:
                #     # samples = make_sample(batch_size, generator, n_sample, dataset)
                #     samples_wo_start = make_sample(batch_size, generator, n_sample, dataset, real_start=False)
                # pd.DataFrame(samples).to_csv(generated_data_path, header=None, index=None)
                # pd.DataFrame(samples_wo_start).to_csv(wo_generated_data_path, header=None, index=None)
                # evaluation(str(dataset), save_name, "gene.csv", f"{save_name}_pretrain_{epoch}")

                # convert the state to latlon            
                # syn_trajectories = []
                # for trajectory in samples_wo_start:
                #     syn_trajectory = []
                #     for state in trajectory:
                #         if state >= dataset.n_locations:
                #             break
                #         lat, lon = grid.state_to_random_latlon_in_the_cell(state)
                #         syn_trajectory.append((0, lat,lon))
                #     syn_trajectories.append(syn_trajectory)

                # save_latlon_with_nan_padding(wo_generated_data_path, syn_trajectories)
                # # save syn_trajectories as csv using pandas with nan paddings that ajdust to the max length of trajectories
                # syn_trajectories = pd.DataFrame(syn_trajectories)
                # syn_trajectories.to_csv(wo_generated_data_path, header=None, index=None)
                
                # original_data_path = "/data/test/test_1/original_data.csv"
                # results = evaluation(original_data_path, wo_generated_data_path, grid)
                # print(results)
                # save results
                # with open(save_path / 'results.json', 'w') as f:
                    # json.dump(results, f)

            # print(results)
            # wo_generated_data_path.parent / f"wo_start_pretrain_{epoch}.csv"

            # evaluation(str(dataset), save_name, "without_real_gene.csv", f"wo_start_{save_name}_pretrain_{epoch}")
            # torch.save(eval_generator.state_dict(), save_path / f"trained_generator_{epoch}.pth")

            # if is_dp:
            #     torch.save(generator._module.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")
            # else:
            #     torch.save(generator.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")

            
            generator.train()

        if early_stopping.early_stop:
            break

        if epsilon > epsilon_for_sgd:
            print("stop because epsilon is larger than epsilon_for_sgd", epsilon, epsilon_for_sgd)
            break

    if args.is_dp:
        with open(save_path / 'epsilnos.json', 'w') as f:
            json.dump(epsilons, f)

    args.end_epoch = early_stopping.epoch
    args.end_loss = early_stopping.best_score
    with open(save_path / "params.json", "w") as f:
        json.dump(vars(args), f)