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
from models import TransGeneratorWithAux, TimeTransGeneratorWithAux, Discriminator, make_sample, GRUNet, Transformer
from evaluation import evaluation
from rollout import Rollout
from loss import GANLoss
from my_utils import EarlyStopping


def pretrain_generator(dataset, generator, save_name, batch_size, n_epochs, generator_lr=2e-3):
    
    cuda_number = next(generator.parameters()).device.index
    
    save_path = (get_datadir() / f"results/{dataset}").parent / f"{save_name}"
    save_path.mkdir(exist_ok=True, parents=True)
    
    kwargs = {'num_workers':1, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    optim_kwargs = {'lr':generator_lr, 'weight_decay':1e-4, 'betas':(.9,.999)}
    optimizer = optim.Adam(generator.parameters(), **optim_kwargs)
    loss_model = nn.NLLLoss(ignore_index=dataset.START_IDX)
    
    print_epoch = 10
    generator.train()
    
    early_stopping = EarlyStopping(patience=20, verbose=True)


    print(f"save model to {save_path}")
    for epoch in tqdm.tqdm(range(n_epochs)):
        loss_value = 0
        for batch in data_loader:
#             print(batch)
            input = batch['input'].reshape(len(batch['input']),-1)
            target = batch['target'].cuda(cuda_number, non_blocking=True).reshape(-1)

            input = input.cuda(cuda_number, non_blocking=True).cuda(cuda_number, non_blocking=True)
#             print(input)
#             print(target)
            
            output = generator(input)
#             print(torch.exp(output))
            output_v = output.view(-1,output.shape[-1])
            
            loss = loss_model(output_v, target)
            loss.backward()

            #apply gradients
            optimizer.step()
            grad_norm = round(generator.embeddings.weight.grad.abs().sum().item(),3)
            optimizer.zero_grad()
            
            loss_value += loss.item()
  
        early_stopping(loss_value / (len(data_loader)), generator)
        if early_stopping.early_stop:
            break

            #print step
        if epoch % print_epoch == 0:
            print('epoch:', epoch, 
                  ' | loss', np.round(loss.item(),2),
                  ' | delta w:', grad_norm)
            
            generated_data_path = save_path/f"gene.csv"
#             real_start = generator.make_initial_data(len(dataset))
#             real_start[:,generator.window_size] = torch.tensor(dataset.data[:, 0])
            print(f"generate to {generated_data_path}")
#             generate_samples(generator, batch_size, dataset.seq_len, len(dataset), generated_data_path, real_start)
            n_sample = len(dataset)
            samples = make_sample(batch_size, generator, n_sample, dataset)
            df = pd.DataFrame(samples).to_csv(generated_data_path, header=None, index=None)
        
            evaluation(str(dataset), save_name, "gene.csv", f"{save_name}_pretrain_{epoch}")
            torch.save(generator.state_dict(), save_path / f"pre_trained_generator_{epoch}.pth")

    torch.save(generator.state_dict(), save_path / f"pre_trained_generator.pth")
#     torch.save()

def pretrain_discriminator(dataset, discriminator, generator, save_name, batch_size, n_epochs):

    cuda_number = next(discriminator.parameters()).device.index
    generated_data_path = save_path/f"gene.csv"
    print(generated_data_path)
    real_data = dataset.data
    generated_num = len(real_data)

    kwargs = {'num_workers':1, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}

    real_start = generator.make_initial_data(generated_num)
#     real_start[:,0] = torch.tensor([generator.start_index]*len(real_data))
    real_start[:,generator.window_size] = torch.tensor(real_data[:, 0])

    dis_criterion = nn.NLLLoss(reduction='mean')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=1e-4)
    dis_criterion = dis_criterion.cuda(cuda_number)

    print_epoch = 10
    print("pretraining discriminator")
    for epoch in tqdm.tqdm(range(n_epochs)):
        generate_samples(generator, args.batch_size, dataset.seq_len, generated_num, generated_data_path, real_start)
        fake_data = pd.read_csv(generated_data_path, header=None).values
        dis_data_iter = make_dis_data_iter(real_data, fake_data, kwargs)

        total_loss = 0
        for data, target in dis_data_iter:
            target = target.cuda(cuda_number)
            data = data.cuda(cuda_number)
            pred = discriminator(data)
            loss = dis_criterion(pred, target)
            total_loss += loss.item()
            dis_optimizer.zero_grad()
            loss.backward()
            dis_optimizer.step()
        total_loss = total_loss / len(dis_data_iter)
        print(total_loss)
        if epoch % print_epoch == 0:
            print('epoch:', epoch)
            torch.save(discriminator.state_dict(), save_path / f"pre_trained_discriminator_{epoch}.pth")
            get_discriminator_score(discriminator, dis_data_iter)

    torch.save(discriminator.state_dict(), save_path / f"pre_trained_discriminator.pth")

        
def get_discriminator_score(discriminator, dis_data_iter):
    
    cuda_number = next(discriminator.parameters()).device.index
    
    targets = []
    preds = []
    for data, target in dis_data_iter:
        target = target.cuda(cuda_number)
        data = data.cuda(cuda_number)
        targets.extend(target.contiguous().view(-1).detach().cpu().numpy())
        preds.extend(torch.exp(discriminator(data)).detach().cpu().numpy())

    preds = [v[1] for v in preds]
    print(preds[:20])
    print("discriminator score:", roc_auc_score(targets, preds))
    
def train(generator, discriminator, dataset, batch_size, save_name, n_epochs, generator_lr, discriminator_lr, dloss_alpha=0):
    
    cuda_number = next(generator.parameters()).device.index
    cuda_number2 = next(discriminator.parameters()).device.index
    
    kwargs = {'num_workers':1, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
    seq_len = dataset.seq_len
    n_vocabs = len(dataset.vocab)
    generated_num = len(dataset)
    
    print("traj_length", seq_len)
    print("vocab_size", n_vocabs)

    real_data = dataset.data
    real_start = generator.make_initial_data(len(real_data))
    real_start[:,0] = torch.tensor([generator.start_index]*len(real_data))
    real_start[:,generator.window_size] = torch.tensor(real_data[:, 0])
    
    save_path = (get_datadir() / f"results/{dataset}").parent / f"{save_name}"
    save_path.mkdir(exist_ok=True, parents=True)

#     logger = get_workspace_logger(save_path)

    print('advtrain generator and discriminator ...')
    update_ratio = 0.8
    rollout = Rollout(generator, update_ratio, cuda_number=cuda_number2)
    rollout_number = 4
    

    gen_gan_loss = GANLoss(cuda_number)
    gen_gan_optm = optim.Adam(generator.parameters(),lr=generator_lr)

    dis_criterion = nn.NLLLoss(reduction='mean')
    dis_optimizer = optim.Adam(discriminator.parameters(),lr=discriminator_lr)
    
    gen_gan_loss = gen_gan_loss.cuda(cuda_number)
    dis_criterion = dis_criterion.cuda(cuda_number2)
    
    generated_data_path = save_path/f"gene.csv"
    print(generated_data_path)
    generate_samples(generator, batch_size, dataset.seq_len, generated_num, generated_data_path, real_start)
#     generate_samples(generator, batch_size, dataset.seq_len, generated_num, save_path/f'gene_epoch_{0}.csv', real_start)
    
    generator.eval()
    discriminator.eval()
    
    evaluation(str(dataset), args.save_name, "gene.csv", f"pretrain")

    fake_data = pd.read_csv(generated_data_path, header=None).values
    dis_data_iter = make_dis_data_iter(real_data, fake_data, kwargs)
    get_discriminator_score(discriminator, dis_data_iter)

    generator.train()
    discriminator.train()
    
    
    print_epoch = 10
    
    if dloss_alpha != 0: 
        M2 = load_M2(str(dataset).split("/")[0], str(dataset).split("/")[1])
        d_crit = distance_loss(dataset, M2, device=next(generator.parameters()).device, window_size=generator.window_size)
        d_crit = d_crit.to(next(generator.parameters()).device)

    with tqdm.tqdm(range(args.resume, n_epochs)) as pbar:
        for epoch in pbar:
                    
            for it in range(1):
                samples = generator.sample(batch_size, start_time=1, data=real_start)
                inputs = samples.contiguous()
#                 print(samples)

#                 rewards = rollout.get_lazy_reward(samples.cuda(cuda_number2), rollout_number, discriminator, generator.window_size)
                rewards = rollout.get_reward(samples.cuda(cuda_number2), rollout_number, discriminator)
                rewards = torch.Tensor(rewards)
                rewards = torch.exp(rewards.cuda(cuda_number)).contiguous().view((-1,))
#                 print(samples)
                print(rewards)
                start_time = np.random.choice(list(range(dataset.seq_len-generator.window_size)))
#                 print(start_time)
                targets = samples.contiguous()[:,start_time+1:start_time+generator.window_size+1].reshape(-1)
#                 print(targets)
                prob = generator.predict_next_location_on_all_stages(inputs, start_time+1)
                gloss = gen_gan_loss(prob, targets, rewards)
                

                if dloss_alpha != 0:
                    dl = d_crit(prob, samples, start_time) * dloss_alpha
                    gloss += dl
                
                gen_gan_optm.zero_grad()
                gloss.backward()
                gen_gan_optm.step()


            rollout.update_params()
            
            for _ in range(4):
                generate_samples(generator, batch_size, dataset.seq_len, generated_num, generated_data_path, real_start)
                fake_data = pd.read_csv(generated_data_path, header=None).values
                dis_data_iter = make_dis_data_iter(real_data, fake_data, kwargs)
                
                total_loss = 0
                for i, (data, target) in enumerate(dis_data_iter):
                    target = target.cuda(cuda_number2)
                    data = data.cuda(cuda_number2)
                    pred = discriminator(data)
                    loss = dis_criterion(pred, target)
                    total_loss += loss.item()
                    dis_optimizer.zero_grad()
                    loss.backward()
                    dis_optimizer.step()
                total_loss = total_loss / (i + 1)

            print('Epoch [%d] Generator Loss: %f, Discriminator Loss: %f' %
                        (epoch, gloss.item(), total_loss))

            results = {}
            
            if dloss_alpha != 0:
                results["distance loss"] = dl.item()
                results["adv g loss"] = gloss.item()-results["distance loss"]
            else:
                results["adv g loss"] = gloss.item()
            results["d loss"] = total_loss
            pbar.set_postfix(results)
            
            if (epoch) % print_epoch == 0:
                torch.save(generator.state_dict(), save_path / f'trained_gen_{epoch}.pth')
                torch.save(discriminator.state_dict(), save_path / f'trained_dis_{epoch}.pth')
                
                generator.eval()
                discriminator.eval()
                
                generate_samples(generator, batch_size, dataset.seq_len, generated_num, generated_data_path, real_start)
                evaluation(str(dataset), save_name, "gene.csv", f"{save_name}_{epoch}")

                fake_data = pd.read_csv(generated_data_path, header=None).values
                dis_data_iter = make_dis_data_iter(real_data, fake_data, kwargs)
                get_discriminator_score(discriminator, dis_data_iter)
                
                generator.train()
                discriminator.train()
                
def make_dis_data_iter(real_data, fake_data, kwargs={}):
#     self.data = torch.concat([real_data, fake_data]).to(device)
#     self.labels = torch.tensor([1]*len(real_data) + [0]*(len(fake_data)))
    dataset = RealFakeDataset(real_data, fake_data)
    data_loader = torch.utils.data.DataLoader(dataset, **kwargs)
    return data_loader
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_number',  default=0, type=int)
    parser.add_argument('--cuda_number2',  default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--dloss_alpha', default='0', type=float)
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--data_name', default='test_1', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--window_size', default=0, type=int)
    parser.add_argument('--n_epochs', default=1000, type=int)
    parser.add_argument('--dpre_epochs', default=20, type=int)
    parser.add_argument('--gpre_epochs', default=120, type=int)
    parser.add_argument('--discriminator_embedding_dim', default=64, type=int)
    parser.add_argument('--generator_embedding_dim', default=128, type=int)
    parser.add_argument('--save_name', default="", type=str)
    parser.add_argument('--generator_lr', default=1e-4, type=float)
    parser.add_argument('--discriminator_lr', default=1e-5, type=float)
    parser.add_argument('--without_M1', action='store_true')
    parser.add_argument('--without_M2', action='store_true')
    parser.add_argument('--without_time', action='store_true')
    parser.add_argument('--gru', action='store_true')
    parser.add_argument('--transformer', action='store_true')
    
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
    
    print("window size:", args.window_size)
    
    if args.without_M1:
        M1 = None
    else:
        M1 = load_M1(dataset)
    
    if args.without_M2:
        M2 = None
    else:
        M2 = load_M2(dataset)
    
    print(M1)
    print(M2)
    
    if args.without_time:
        print("without time embedding")
        generator = TransGeneratorWithAux(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim, M1, M2).cuda(args.cuda_number)
    else:
        generator = TimeTransGeneratorWithAux(n_vocabs, args.window_size, dataset.seq_len, dataset.START_IDX, dataset.MASK_IDX, dataset.CLS_IDX, args.generator_embedding_dim, M1, M2).cuda(args.cuda_number)
    generator.real = False
    
    if args.gru:
        print("USE GRU")
        input_dim = dataset.n_locations+1
        output_dim = dataset.n_locations
        hidden_dim = 256
        n_layers = 2
        generator = GRUNet(input_dim, hidden_dim, output_dim, n_layers).cuda(args.cuda_number)
        
    if args.transformer:
        print("USE TRANSFORMER")
        embed_size = args.generator_embedding_dim
        inner_ff_size = embed_size*4
        n_heads = 8
        n_code = 8
        generator = Transformer(n_code, n_heads, embed_size, inner_ff_size, n_vocabs, dataset.n_locations, dataset.seq_len, dataset.CLS_IDX, 0.1).cuda(args.cuda_number)
        
    
    discriminator = Discriminator(dataset.seq_len, total_locations=n_vocabs, embedding_dim=args.discriminator_embedding_dim).cuda(args.cuda_number2)
    

    if args.resume != 0:
        print(f"loaded from {save_path / f'trained_gen_{args.resume}.pth'}")
        print(f"loaded from {save_path / f'trained_dis_{args.resume}.pth'}")
        generator.load_state_dict(torch.load(save_path / f'trained_gen_{args.resume}.pth'))
        discriminator.load_state_dict(torch.load(save_path / f'trained_dis_{args.resume}.pth'))
    else:
#         if (save_path / 'pre_trained_generator.pth').exists():
        if False:
            print("skip pretraining generator")
            generator.load_state_dict(torch.load(save_path / f'pre_trained_generator.pth'))
            print(f"loaded from {save_path / 'pre_trained_generator.pth'}")
        else:
            print("pretraining generator")
            pretrain_generator(dataset, generator, args.save_name, args.batch_size, args.gpre_epochs)

        if (save_path / 'pre_trained_discriminator.pth').exists():
            print("skip pretraining discriminator")
            discriminator.load_state_dict(torch.load(save_path / f'pre_trained_discriminator.pth'))
            print(f"loaded from {save_path / 'pre_trained_discriminator.pth'}")
        else:
            print("pretrain discriminator")
            pretrain_discriminator(dataset, discriminator, generator, args.save_name, args.batch_size, args.dpre_epochs)

    
    train(generator, discriminator, dataset, args.batch_size, args.save_name, args.n_epochs, args.generator_lr, args.discriminator_lr, args.dloss_alpha)