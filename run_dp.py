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


from my_utils import load_latlon_range, make_gps, load_M1, load_M2, get_datadir, load_dataset, generate_samples, get_maxdistance
from dataset import RealFakeDataset
from models import TransGenerator_DP, Discriminator
from evaluation import evaluation
from rollout import Rollout
from loss import GANLoss

from opacus import PrivacyEngine


def train(generator, dataset, save_name, n_epochs, batch_size, lr):
    
    cuda_number = next(generator.parameters()).device.index
    
    kwargs = {'num_workers':1, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    
    optim_kwargs = {'lr':lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
    optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
    loss_model = nn.NLLLoss(ignore_index=dataset.START_IDX)
    
    print_epoch = 10
    
    privacy_engine = PrivacyEngine()
    generator, optimizer, data_loader = privacy_engine.make_private(
        module=generator,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    
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

        if epoch % print_epoch == 0:
            print('epoch:', epoch, 
                  ' | loss', np.round(loss.item(),2),
                  ' | delta w:', grad_norm)
            
            generated_data_path = save_path/f"gene.csv"
            real_start = generator.make_initial_data(len(dataset))
            real_start[:,generator.window_size] = torch.tensor(dataset.data[:, 0])
            print(f"generate to {generated_data_path}")
            generate_samples(generator, batch_size, dataset.seq_len, len(dataset), generated_data_path, real_start)
            evaluation(str(dataset), save_name, "gene.csv", f"{save_name}_pretrain_{epoch}")
            torch.save(generator.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")

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

    generator = TransGenerator_DP(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim).cuda(args.cuda_number)
    generator.real = False
    
    train(generator, dataset, args.save_name, args.n_epochs, args.batch_size, args.lr)