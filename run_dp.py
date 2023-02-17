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
from sklearn.metrics import roc_auc_score


from my_utils import load_latlon_range, make_gps, load_M1, load_M2, get_datadir, load_dataset, get_maxdistance
from dataset import RealFakeDataset
from models import Discriminator, Transformer, GRUNet, DPGRUNet, make_sample
from evaluation import evaluation
from rollout import Rollout
from loss import GANLoss

from opacus import PrivacyEngine


def train(generator, dataset, save_name, n_epochs, batch_size, lr, is_dp=False):
    
    cuda_number = next(generator.parameters()).device.index
    
    kwargs = {'num_workers':1, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    optim_kwargs = {'lr':lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
    optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
    loss_model = nn.NLLLoss(ignore_index=dataset.START_IDX)
    
    print_epoch = 10
    
    if is_dp:
        privacy_engine = PrivacyEngine()
        generator, optimizer, data_loader = privacy_engine.make_private(
            module=generator,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

    delta = 1e-5
    
    epsilons = []
    
    for epoch in tqdm.tqdm(range(n_epochs)):
        for batch in data_loader:
            input = batch['input'].reshape(len(batch['input']),-1)
            target = batch['target'].cuda(cuda_number, non_blocking=True).reshape(-1)

            input = input.cuda(cuda_number, non_blocking=True).cuda(cuda_number, non_blocking=True)
            
            output = generator(input)
            output_v = output.view(-1,output.shape[-1])
            
            loss = loss_model(output_v, target)
            loss.backward()

            optimizer.step()
            grad_norm = round(generator.embeddings.weight.grad.abs().sum().item(),3)
            optimizer.zero_grad()

        epsilon = privacy_engine.get_epsilon(delta)
        epsilons.append(epsilon)
        print(f'epsilon{epsilon}, delta{delta}')
        if epoch % print_epoch == 0:
            
            generator.eval()
            print('epoch:', epoch, 
                  ' | loss', np.round(loss.item(),2),
                  ' | delta w:', grad_norm)
            
            generated_data_path = save_path/f"gene.csv"
            wo_generated_data_path = save_path/f"without_real_gene.csv"
            # real_start = generator.make_initial_data(len(dataset))
            # real_start[:,generator.window_size] = torch.tensor(dataset.data[:, 0])
            # print(f"generate to {generated_data_path}")
            # generate_samples(generator, batch_size, dataset.seq_len, len(dataset), generated_data_path, real_start)
            n_sample = len(dataset)
            samples = make_sample(batch_size, generator, n_sample, dataset)
            samples_wo_start = make_sample(batch_size, generator, n_sample, dataset, real_start=False)
            pd.DataFrame(samples).to_csv(generated_data_path, header=None, index=None)
            pd.DataFrame(samples_wo_start).to_csv(wo_generated_data_path, header=None, index=None)
            evaluation(str(dataset), save_name, "gene.csv", f"{save_name}_pretrain_{epoch}")
            evaluation(str(dataset), save_name, "without_real_gene.csv", f"wo_start_{save_name}_pretrain_{epoch}")
            torch.save(generator._module.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")
            
            generator.train()
            
    with open(save_path / 'epsilnos.json', 'w') as f:
        json.dump(epsilons, f)

    torch.save(generator.state_dict(), save_path / f"pre_trained_generator.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number',  default=0, type=int)
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--data_name', default='test_1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--window_size', default=0, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--generator_embedding_dim', default=128, type=int)
    parser.add_argument('--save_name', default="", type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--is_dp', action='store_true')
    
    args = parser.parse_args()
    
    if args.save_name == "":
        args.save_name = args.data_name
        
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    data_dir = get_datadir()
    
    save_path = data_dir / "results" / args.dataset / args.save_name
    data_path = data_dir / f"/{args.dataset}" / f"{args.data_name}"
    
    save_path.mkdir(exist_ok=True, parents=True)
    with open(save_path / "param.json", "w") as f:
        json.dump(vars(args), f)
    
    dataset = load_dataset(args.dataset, args.data_name, args.window_size)
    args.window_size = dataset.window_size
    n_vocabs = len(dataset.vocab)

    # generator = TransGenerator_DP(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)
    # generator = Transformer(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)
    
    if args.gru:
        input_dim = dataset.n_locations+1
        output_dim = dataset.n_locations
        hidden_dim = 256
        n_layers = 2
        if args.is_dp:
            generator = DPGRUNet(input_dim, hidden_dim, output_dim, n_layers).cuda(args.cuda_number)
        else:
            generator = GRUNet(input_dim, hidden_dim, output_dim, n_layers).cuda(args.cuda_number)
    else:
        embed_size = args.generator_embedding_dim
        inner_ff_size = embed_size*4
        n_heads = 8
        n_code = 8
        generator = Transformer(n_code, n_heads, embed_size, inner_ff_size, n_vocabs, dataset.n_locations, dataset.seq_len, dataset.CLS_IDX, 0.1).cuda(args.cuda_number)
        # generator.real = False

    train(generator, dataset, args.save_name, args.n_epochs, args.batch_size, args.lr, args.is_dp)