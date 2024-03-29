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

from my_utils import load_latlon_range, make_gps, load_M1, load_M2, get_datadir, load_dataset
from dataset import RealFakeDataset, TrajectoryDataset, TrajectorySelfAttentionDataset, padded_collate_without_end, make_padded_collate, int_to_float_of_minute, convert_time_traj_to_time_traj_float, TrajectoryDataset_with_Time, traj_to_format
from models import Discriminator, Transformer, SelfAttentionTransformer, TimeGRUNet, GRUNet, DPGRUNet, make_sample, make_sample_with_time, make_sample_with_time_and_traj_type, TimeTrajTypeGRUNet
from evaluation import evaluation
from rollout import Rollout
from loss import GANLoss
from grid import Grid
from data_processing import save_latlon_with_nan_padding, save_state_with_nan_padding

from opacus import PrivacyEngine

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
    
def train_with_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels):
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
    loss_location = loss_model(output_locations_v, target_locations.view(-1))
    loss_time = loss_time_fn(output_times_v, target_times, ignore_value=dataset.IGNORE_IDX)
    loss = loss_location + loss_time
    loss.backward()

    # WARNING!!!!!
    # I think it's ok, but it's possible that this includes bugs because of the gradient accumulation
    # print(generator.embeddings.weight.grad_sample.shape)
    # print(optimizer.grad_samples.shape)

    # count the number of location in each trajectory
    counts = []
    for input_location in input_locations:
        counts.append(sum(input_location != dataset.IGNORE_IDX))

    # divide gradients by training length (except for ignore_value)
    for param in generator.parameters():
        for i in range(len(counts)):
            param.grad_sample[i] /= counts[i]

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


# def train(generator, dataset, save_name, n_epochs, batch_size, lr, is_dp=False):
    
#     cuda_number = next(generator.parameters()).device.index
    
#     kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
#     data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
#     optim_kwargs = {'lr':lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
#     optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
#     # loss_model = nn.NLLLoss(ignore_index=dataset.START_IDX)
#     loss_model = nn.NLLLoss()
    
#     print_epoch = 10
    
#     dataset.data_loader = data_loader
#     if is_dp:
#         privacy_engine = PrivacyEngine()
#         generator, optimizer, data_loader = privacy_engine.make_private(
#             module=generator,
#             optimizer=optimizer,
#             data_loader=data_loader,
#             noise_multiplier=1.1,
#             max_grad_norm=1.0,
#         )
#         delta = 1e-5
#         epsilons = []

#     for epoch in tqdm.tqdm(range(n_epochs)):
#         # reset the dataset of the fixed sequence length
#         dataset.reset()

#         for batch in data_loader:
#             input = batch['input'].reshape(len(batch['input']),-1)
#             target = batch['target'].cuda(cuda_number, non_blocking=True).reshape(-1)

#             input = input.cuda(cuda_number, non_blocking=True).cuda(cuda_number, non_blocking=True)
            
#             # print(input)
#             # print(target)
#             output = generator(input)
#             output_v = output.view(-1,output.shape[-1])

#             loss = loss_model(output_v, target)
#             loss.backward()

#             print("Probably, this includes bags because each record has multiple gradients")

#             optimizer.step()
#             grad_norm = round(generator.embeddings.weight.grad.abs().sum().item(),3)
#             optimizer.zero_grad()

#         if is_dp:
#             epsilon = privacy_engine.get_epsilon(delta)
#             epsilons.append(epsilon)
#             print(f'epsilon{epsilon}, delta{delta}')

    
#         if epoch % print_epoch == 0:
            
#             generator.eval()
#             print('epoch:', epoch, 
#                   ' | loss', np.round(loss.item(),2),
#                   ' | delta w:', grad_norm)
            
#             # generated_data_path = save_path/f"gene.csv"
#             wo_generated_data_path = save_path/f"without_real_gene.csv"
#             # real_start = generator.make_initial_data(len(dataset))
#             # real_start[:,generator.window_size] = torch.tensor(dataset.data[:, 0])
#             # print(f"generate to {generated_data_path}")
#             # generate_samples(generator, batch_size, dataset.seq_len, len(dataset), generated_data_path, real_start)
#             n_sample = len(dataset)


#             if is_dp:
#                 # samples = make_sample(batch_size, generator._module, n_sample, dataset)
#                 samples_wo_start = make_sample(batch_size, generator._module, n_sample, dataset, real_start=False)
#             else:
#                 # samples = make_sample(batch_size, generator, n_sample, dataset)
#                 samples_wo_start = make_sample(batch_size, generator, n_sample, dataset, real_start=False)
#             # pd.DataFrame(samples).to_csv(generated_data_path, header=None, index=None)
#             pd.DataFrame(samples_wo_start).to_csv(wo_generated_data_path, header=None, index=None)
#             # evaluation(str(dataset), save_name, "gene.csv", f"{save_name}_pretrain_{epoch}")

#             evaluation(str(dataset), save_name, "without_real_gene.csv", f"wo_start_{save_name}_pretrain_{epoch}")
#             if is_dp:
#                 torch.save(generator._module.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")
#             else:
#                 torch.save(generator.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")

            
#             generator.train()
            
#     with open(save_path / 'epsilnos.json', 'w') as f:
#         json.dump(epsilons, f)

#     torch.save(generator.state_dict(), save_path / f"pre_trained_generator.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number',  default=0, type=int)
    parser.add_argument('--print_epoch',  default=10, type=int)
    parser.add_argument('--n_generated',  default=0, type=int)
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--data_name', default='test_1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--n_layers', default=1, type=int)
    parser.add_argument('--window_size', default=0, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--save_name', default="", type=str)
    parser.add_argument('--loss_reduction', default="mean", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dp_noise_multiplier', default=1.1, type=float)
    parser.add_argument('--clipping_bound', default=1.0, type=float)
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--is_dp', action='store_true')
    parser.add_argument('--is_traj_type', action='store_true')
    parser.add_argument('--is_time', action='store_true')
    parser.add_argument('--is_self_attention', action='store_true')
    parser.add_argument('--is_evaluation', action='store_true')
    parser.add_argument('--without_end', action='store_true')
    parser.add_argument('--only_first_markov', action='store_true')
    parser.add_argument('--is_test_bias', action='store_true')

    args = parser.parse_args()
    
    if args.save_name == "":
        args.save_name = args.data_name
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = get_datadir()
    
    save_path = data_dir / "results" / args.dataset / args.save_name
    data_path = data_dir / f"{args.dataset}" / f"{args.data_name}"
    
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / "param.json", "w") as f:
        json.dump(vars(args), f)
    
    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    lat_range = param["lat_range"]
    lon_range = param["lon_range"]

    data_name = f"training_data_0000.csv"
    if (data_path / data_name).exists():
        # load data from data_path training_data_*.csv
        data_dirs = glob.glob(str(data_path / "training_data_0*.csv"))
        # sort data_dirs
        data_dirs = sorted(data_dirs)
        for data_dir in data_dirs:
            print(f"load from {data_dir}")
        # data_dirs = []
        # for i in range(28):
        #     data_name = f"training_data_{i:04d}.csv"
        #     data_dirs.append(data_path / data_name)
        #     print(f"load from {data_path / data_name}")
        time_data_dirs = []
        for i in range(len(data_dirs)):
            time_data_dirs.append(str(data_path / f"training_data_time_{i:04d}.csv"))
        for time_data_dir in time_data_dirs:
            print(time_data_dir)

        real_time_traj = load_dataset(time_data_dirs)
        trajectories = load_dataset(data_dirs)

        # remove trajs including None
        # remove time trajs whose index is None in trajectories
        real_time_traj = [real_time_traj[i] for i in range(len(trajectories)) if "None" not in trajectories[i]]
        trajectories = [trajectory for trajectory in trajectories if "None" not in trajectory]
        real_time_traj = [[(float(v.split("_")[0]), float(v.split("_")[1])) for v in trajectory if v != 'nan'] for trajectory in real_time_traj]
        trajectories = [[int(float(v)) for v in trajectory if v != 'nan'] for trajectory in trajectories]

        print(f"len of trajectories: {len(trajectories)}")
        print(f"len of real_time_traj: {len(real_time_traj)}")

        label_count, format_to_label, label_to_format = compute_traj_type_distribution(trajectories)

    else:
        print("make training data")
        make_training_data(data_path, n_bins)

    # original_data_path = data_path / "original_data.csv"

    # compute max seq len
    max_seq_len = max([len(trajectory) for trajectory in trajectories])
    print(f"max seq len: {max_seq_len}")
    
    if args.window_size == 0:
        args.window_size = max_seq_len
        print(f"window size is set as {args.window_size}")

    # make fake format_to_label
    fake_format_to_label = True
    if fake_format_to_label:
        print("WARNING: fake format_to_label is used")
        for key in format_to_label.keys():
            format_to_label[key] = 0


    if args.is_time:
        dataset = TrajectoryDataset_with_Time(trajectories, real_time_traj, n_bins, format_to_label)
    elif args.is_self_attention:
        dataset = TrajectorySelfAttentionDataset(data, args.window_size, max_seq_len, n_bins, f"{args.dataset}/{args.data_name}", random_mask=False)
    else:
        # dataset = TrajectoryDataset(trajectories, args.window_size, max_seq_len, n_bins, f"{args.dataset}/{args.data_name}", random_mask=False)
        dataset = TrajectoryDataset(trajectories, n_bins)
    n_vocabs = len(dataset.vocab)

    # ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    # grid = Grid(ranges)

    # generator = TransGenerator_DP(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)
    # generator = Transformer(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)

    dataset_labels = torch.tensor([format_to_label[traj_to_format(trajectory)] for trajectory in trajectories])

    if args.gru:
        input_dim = dataset.n_locations+2
        output_dim = dataset.n_locations+1
        hidden_dim = args.hidden_dim
        embed_size = args.embed_dim
        n_layers = args.n_layers
        traj_type_dim = len(label_count)
        if args.is_traj_type:
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
        kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':make_padded_collate(dataset.IGNORE_IDX)}
    elif args.only_first_markov:
        print("only first markov")
        kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':padded_collate_without_end_only_first_markov}
    elif args.without_end:
        kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':padded_collate_without_end}
    else:    
        kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size, 'collate_fn':padded_collate}
    # kwargs = {'num_workers':0, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':args.batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    optim_kwargs = {'lr':args.lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
    optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
    # loss_model = nn.NLLLoss(ignore_index=dataset.START_IDX)
    print("IGNORE", dataset.IGNORE_IDX)
    loss_model = nn.NLLLoss(ignore_index=dataset.IGNORE_IDX)
    
    print_epoch = args.print_epoch
    
    dataset.data_loader = data_loader
    if args.is_dp:
        print("IS DP")
        privacy_engine = PrivacyEngine(accountant='rdp')
        generator, optimizer, data_loader_ = privacy_engine.make_private(
            module=generator,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=args.dp_noise_multiplier,
            max_grad_norm=args.clipping_bound,
            loss_reduction=args.loss_reduction
        )
        delta = 1e-5
        epsilons = []

    if args.is_dp:
        eval_generator = generator._module
    else:
        eval_generator = generator


    for epoch in tqdm.tqdm(range(args.n_epochs)):
        # reset the dataset of the fixed sequence length
        dataset.reset()
        loss = 0
        for i, batch in enumerate(data_loader):
            if len(batch["input"]) == 0:
                continue
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
                    loss += train_with_time(generator, optimizer, loss_model, input_locations, target_locations, input_times, target_times, labels)
            else:
                loss += train(generator, optimizer, loss_model, input, target)

        if args.is_dp:
            epsilon = privacy_engine.get_epsilon(delta)
            epsilons.append(epsilon)
            print(f'epsilon{epsilon}, delta{delta}, loss {loss/(i+1)}')

        if epoch != 0 and epoch % print_epoch == 0:
            
            generator.eval()
            print('epoch:', epoch, ' | loss', loss/(i+1))

            if args.is_evaluation:
                
                generated_data_path = save_path/f"gene_{epoch}.csv"
                generated_time_data_path = save_path/f"gene_time_{epoch}.csv"
                # wo_generated_data_path = save_path / f"without_real_gene.csv"
                # real_start = generator.make_initial_data(len(dataset))
                # real_start[:,generator.window_size] = torch.tensor(dataset.data[:, 0])
                # print(f"generate to {generated_data_path}")
                # generate_samples(generator, batch_size, dataset.seq_len, len(dataset), generated_data_path, real_start)
                if args.n_generated == 0:
                    args.n_generated = len(dataset)
                # print(dataset_labels)
                # n_sample = 
                if args.is_traj_type:
                    # print(dataset_labels)
                    # dataset_labels = torch.Tensor([0]*args.n_generated).long()
                    dataset_labels = torch.Tensor(np.random.choice(dataset_labels, args.n_generated, replace=True)).long()
                    print(dataset_labels)
                    # print(dataset_labels)
                    samples, time_samples = make_sample_with_time_and_traj_type(args.batch_size, eval_generator, dataset_labels, dataset, label_to_format)
                elif args.is_time:
                    samples, time_samples = make_sample_with_time(args.batch_size, eval_generator, args.n_generated, dataset)
                else:
                    samples = make_sample(args.batch_size, eval_generator, args.n_generated, dataset)

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
            torch.save(eval_generator.state_dict(), save_path / f"trained_generator_{epoch}.pth")

            # if is_dp:
            #     torch.save(generator._module.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")
            # else:
            #     torch.save(generator.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")

            
            generator.train()
            
    with open(save_path / 'epsilnos.json', 'w') as f:
        json.dump(epsilons, f)

    torch.save(generator.state_dict(), save_path / f"pre_trained_generator.pth")

    # train(generator, dataset, args.save_name, args.n_epochs, args.batch_size, args.lr, args.is_dp)