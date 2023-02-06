# coding: utf-8
import sys
sys.path.append('../')

import pdb              
import math
import torch
import bisect
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from utils import *
from src.my_utils import load_M1, load_M2
from my_model.model import Transformer


def gen_gaussian_dist(sigma=10):
    """Return a single-sided gaussian distribution weight array and its index.
    """
    u = 0
    x = np.linspace(0, 1, 100)
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / \
        (math.sqrt(2 * math.pi) * sigma)
    return x, y


class TransGenerator(Transformer):
    
    def __init__(self, n_vocabs, window_size, seq_len, start_index, mask_index, cls_index, generator_embedding_dim):
#         embed_size = 128
        embed_size = generator_embedding_dim
        inner_ff_size = embed_size * 4
        n_heads = 8
        n_code = 8
        super().__init__(n_code, n_heads, embed_size, inner_ff_size, n_vocabs, window_size+1, 0.1)
        self.start_index = start_index 
        self.mask_index = mask_index
        self.cls_index = cls_index
        self.seq_len = seq_len
#         self.cuda_number = cuda_number
        self.n_vocabs = n_vocabs
        self.n_locations = n_vocabs - 5
        self.window_size = window_size

    def make_initial_data(self, n_sample):
        samples = torch.tensor([[self.start_index]*(self.seq_len+self.window_size)]*n_sample).long()
        samples[:,0] = self.start_index
#         if self.real_start:
#             samples[:,1] = self.real_start
            
        return samples
        
#     def make_input_for_predict_next_location_on_all_stages(self, x):
#         input = []
#         for traj in x:
#             for i in range(self.seq_len):
#                 input.append([self.start_index] + [state.item() for state in traj[:i]] + [self.mask_index] * (self.seq_len - i))
#         return torch.tensor(input).long()

    def make_input_for_predict_next_location_on_all_stages(self, x, start_time=0):
        input = []
        for traj in x:
            for i in range(self.window_size):
#                 input.append([self.start_index]*(self.window_size-start_time-i) + [state.item() for state in traj[max(start_time-self.window_size+i,0):start_time+i]] + [self.cls_index])
                input.append([self.start_index]*(self.window_size-start_time-i) + [state.item() for state in traj[max(start_time-self.window_size+i,0):start_time+i]])
                
        return torch.tensor(input).long()
                
       
    def predict_next_location_on_all_stages(self, x, start_time=0):
#         input = self.make_input_for_predict_next_location_on_all_stages(x).to(next(self.parameters()).device)
        input = self.make_input_for_predict_next_location_on_all_stages(x, start_time).to(next(self.parameters()).device)
#         print(input.shape)
        probs = []
        for i in range(int(len(input)/self.window_size)):
#             prob = self(input[i*self.seq_len:(i+1)*self.seq_len])
#             print(i*self.window_size, (i+1)*self.window_size)
#             cls_array = torch.tensor([self.cls_index]*(len(input))).reshape(-1,1).to(next(self.parameters()).device)
#             print(windowed_input.shape)
            windowed_input = input[i*self.window_size:(i+1)*self.window_size].to(next(self.parameters()).device)
#             print(windowed_input)
#             print(windowed_input.shape, cls_array.shape)
#             windowed_input = torch.concat([windowed_input, cls_array], dim=1)
#             print(windowed_input)
            prob = self(windowed_input)
            
#             prob = self(input[i*self.window_size:(i+1)*self.window_size])[:,-1,:self.n_locations]
#             print("a", prob.shape)
#             print(self.seq_len)
#             for j in range(self.seq_len):
#                 print(j)
#                 print(prob[j,j+1])
#                 print(prob.shape)
            probs.append(prob)
        return torch.concat(probs).reshape(x.shape[0]*self.window_size, -1)
    
        
#     def predict_next_location_on_all_stages(self, x):
#         input = self.make_input_for_predict_next_location_on_all_stages(x).to(next(self.parameters()).device)
# #         print(input.shape)
#         probs = []
#         for i in range(int(len(input)/self.seq_len)):
# #             prob = self(input[i*self.seq_len:(i+1)*self.seq_len])
#             prob = self.predict(input[i*self.seq_len:(i+1)*self.seq_len], range(1,self.seq_len+1))
# #             print("a", prob.shape)
# #             print(self.seq_len)
# #             for j in range(self.seq_len):
# #                 print(j)
# #                 print(prob[j,j+1])
# #                 print(prob.shape)
#             probs.append(prob)
#         return torch.concat(probs).reshape(x.shape[0]*x.shape[1], -1)

    def step(self, i, sample):
        
        if self.real == True:
            self.step_from_real(i, sample)
            return
        
        input = sample.to(next(self.parameters()).device)[:, i:self.window_size+i]
        probs = torch.exp(self(input)).detach().cpu().numpy()
#         probs = self(input).detach().cpu().numpy()
#         print(probs[0])
#         print("b", np.where(probs[0] != 0))
        
        for j, prob in enumerate(probs):
#             print(prob.sum())
            prob = prob / prob.sum()
#             print(prob)
            sample[j, self.window_size+i] = np.random.choice(self.n_locations, p=prob)
        return sample
    
    
    def step_from_real(self, i, sample):
        input = sample.to(next(self.parameters()).device)[:, i:self.window_size+i]
        probs = torch.exp(self(input)).detach().cpu().numpy()
        real = self.data[:, i]
        multiplier = np.zeros(self.n_locations)
        for v in real:
            if self.sum == True:
                multiplier[v] += 1
            else:
                multiplier[v] = 1
            
        for j, prob in enumerate(probs):
#             print(multiplier)
            prob *= multiplier
            prob = prob / prob.sum()
            sample[j, self.window_size+i] = np.random.choice(self.n_locations, p=prob)
    
        
    def sample(self, n_sample, start_time=0, data=None):
        
        if data is None:
            sample = self.make_initial_data(n_sample)
        else:
            sample = data.detach().cpu()
            indice = np.random.choice(range(len(data)), n_sample, replace=False)
            sample = sample[indice]
#         print(sample)
        for i in range(start_time, self.seq_len):
            self.step(i, sample)
#             input = sample.to(next(self.parameters()).device)[:, i:self.window_size+i]
# #             print(input)
# #             cls_array = torch.tensor([self.cls_index]*(len(input))).reshape(-1,1).to(next(self.parameters()).device)
# #             input = torch.concat([input, cls_array], dim=1)
# #             probs = torch.exp(self(input)[:,-1,:self.n_locations]).detach().cpu().numpy()
#             probs = torch.exp(self(input)).detach().cpu().numpy()
# #             probs = torch.exp(self.predict(input, [i+1]*len(input))).detach().cpu().numpy()
#             for j, prob in enumerate(probs):
#                 prob = prob / prob.sum()
#                 sample[j, self.window_size+i] = np.random.choice(self.n_locations, p=prob)
        return sample[:,self.window_size:]
    
    def forward_without_softmax(self, x):
        cls_array = torch.tensor([self.cls_index]*(len(x))).reshape(-1,1).to(next(self.parameters()).device)
        x = torch.concat([x, cls_array], dim=1)
#         mask = torch.ones(x.shape).to(next(self.parameters()).device)
#         mask[x==self.start_index] = 0
        mask = None
#         print(mask)
        x = super().forward(x, mask=mask)[:,-1,:self.n_locations]
    
    def forward(self, x):
        x = self.forward_without_softmax(x)
        
        return x
#     def predict(self, x, predict_places):
#         x = self(x)
        
#         predicts = []
#         for i, predict_place in enumerate(predict_places):
#             predicts.append(x[i,predict_place,:self.n_locations])
#         x = torch.concat(predicts).reshape(-1, self.n_locations)
        
#         return x

    
    
class TransGeneratorWithAux(TransGenerator):
    
    def __init__(self, n_vocabs, window_size, seq_len, start_index, mask_index, cls_index, generator_embedding_dim, M1):
        super().__init__(n_vocabs, window_size, seq_len, start_index, mask_index, cls_index, generator_embedding_dim)

        self.M1 = M1
        self.M1 = np.concatenate([M1, np.zeros((1,M1.shape[1]))], axis=0)
        self.linear_dim = 128
        self.linear_mat1 = nn.Linear(self.n_locations, self.linear_dim)
        self.linear_mat1_2 = nn.Linear(self.linear_dim, self.n_locations)
        

    def forward(self, x):
        batch_size = len(x)
        last_locs = x[:, -1].cpu().detach().numpy()
#         print(last_locs)
            
        mat1 = self.M1[last_locs]
        a = last_locs[0]
        mat1 = torch.Tensor(mat1).to(next(self.parameters()).device)
#         mat1 = (mat1 == 0)
#         minus_infty = (torch.ones(mat1.shape) * -1e10).to(next(self.parameters()).device)
        
        mat1 = F.relu(self.linear_mat1(mat1))
        mat1 = torch.sigmoid(self.linear_mat1_2(mat1))
        mat1 = F.normalize(mat1)
#         print(x[0])
#         print("a", torch.where(mat1[0]))
        x = super().forward(x)
#         x[mat1] = -1e10
#         x = x * mat1 * minus_infty
#         print(x.shape, mat1.shape)
#         x = F.log_softmax(x, dim=-1)
#         print(x.shape)
        x = x + torch.mul(x,mat1)
        x = F.log_softmax(x, dim=-1)
        return x
    
    

class Generator(nn.Module):
    """Basic Generator.
    """

    def __init__(
            self,
            total_locations=8606,
            embedding_net=None,
            embedding_dim=32,
            hidden_dim=64,
            bidirectional=False,
            cuda=None,
            starting_sample='zero',
            starting_dist=None):
        """

        :param total_locations:
        :param embedding_net:
        :param embedding_dim:
        :param hidden_dim:
        :param bidirectional:
        :param cuda:
        :param starting_sample:
        :param starting_dist:
        """
        super(Generator, self).__init__()
        self.total_locations = total_locations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.linear_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.use_cuda = cuda
        self.starting_sample = starting_sample
        if self.starting_sample == 'real':
            self.starting_dist = torch.tensor(starting_dist).float()

        if embedding_net:
            self.embedding = embedding_net
        else:
            self.embedding = nn.Embedding(
                num_embeddings=total_locations, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(self.linear_dim, total_locations)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def init_hidden(self, batch_size):
        h = Variable(torch.zeros(
            (2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros(
            (2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        if self.use_cuda is not None:
            h, c = h.cuda(), c.cuda()
        return h, c

    def forward(self, x):
        """

        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        h0, c0 = self.init_hidden(x.size(0))
        x, (h, c) = self.lstm(x, (h0, c0))
        pred = F.log_softmax(self.linear(
            x.contiguous().view(-1, self.linear_dim)), dim=-1)
        return pred

    def step(self, x, h, c):
        """

        :param x: (batch_size, 1), current location
        :param h: (1/2, batch_size, hidden_dim), lstm hidden state
        :param c: (1/2, batch_size, hidden_dim), lstm cell state
        :return:
            (batch_size, total_locations), prediction of next stage
        """
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        x, (h, c) = self.lstm(x, (h, c))
        pred = F.softmax(self.linear(x.view(-1, self.linear_dim)), dim=-1)
        return pred, h, c

    def sample(self, batch_size, seq_len, x=None):
        """

        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        res = []
        flag = False  # whether sample from zero
        if x is None:
            flag = True
        s = 0
        if flag:
            if self.starting_sample == 'zero':
                x = Variable(torch.zeros((batch_size, 1)).long())
            elif self.starting_sample == 'rand':
                x  = Variable(torch.randint(
                        high=self.total_locations, size=(batch_size, 1)).long())
            elif self.starting_sample == 'real':
                x = Variable(torch.stack(
                    [torch.multinomial(self.starting_dist, 1) for i in range(batch_size)], dim=0))
                s = 1
        self.lstm.flatten_parameters()
        if self.use_cuda is not None:
            x = x.cuda()
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                x, h, c = self.step(x, h, c)
                x = x.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                x, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = x.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                x, h, c = self.step(x, h, c)
                x = x.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output


            


class ATGenerator(nn.Module):
    """Attention Generator.
    """

    def __init__(
            self,
            total_locations=8606,
            embedding_net=None,
            loc_embedding_dim=256,
            tim_embedding_dim=16,
            hidden_dim=64,
            bidirectional=False,
            data='Geolife',
            device=None,
            function=False, 
            starting_sample='zero',
            starting_dist=None,
            name=""):
        """

        :param total_locations:
        :param embedding_net:
        :param embedding_dim:
        :param hidden_dim:
        :param bidirectional:
        :param cuda:
        :param starting_sample:
        :param starting_dist:
        """
        super(ATGenerator, self).__init__()
        self.total_locations = total_locations
        self.loc_embedding_dim = loc_embedding_dim
        self.tim_embedding_dim = tim_embedding_dim
        self.embedding_dim = loc_embedding_dim + tim_embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.linear_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.device = device
        self.data = data
        self.starting_sample = starting_sample
        self.function = function
        # process distance weights
        # self.M1 = np.load('../data/%s/M1.npy' % self.data)
        # self.M2 = np.load('../data/%s/M2.npy' % self.data)
        self.M1 = load_M1(data, name)
        self.M2 = load_M2(data, name)

        
        if self.starting_sample == 'real':
            self.starting_dist = torch.tensor(starting_dist).float()

        if embedding_net:
            self.embedding = embedding_net
        else:
            self.loc_embedding = nn.Embedding(
                num_embeddings=self.total_locations, embedding_dim=self.loc_embedding_dim)
            self.tim_embedding = nn.Embedding(
                num_embeddings=24, embedding_dim=self.tim_embedding_dim) 
        
        
        self.attn = nn.MultiheadAttention(self.hidden_dim,  4)
        self.Q = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.V = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.K = nn.Linear(self.embedding_dim, self.hidden_dim)
        
        self.attn2 = nn.MultiheadAttention(self.hidden_dim, 1)
        self.Q2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.linear_dim, self.total_locations)
        self.linear_mat1 = nn.Linear(self.total_locations, self.linear_dim)
        self.linear_mat1_2 = nn.Linear(self.linear_dim, self.total_locations)

        self.linear_mat2 = nn.Linear(self.total_locations, self.linear_dim)
        self.linear_mat2_2 = nn.Linear(self.linear_dim, self.total_locations)

        self.final_linear = nn.Linear(self.linear_dim, self.total_locations)

        if function:
            self.M3 = np.load('../data/%s/M3.npy' % self.data)
            self.linear_mat3 = nn.Linear(self.total_locations, self.linear_dim)


        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def init_hidden(self, batch_size):
        h = torch.LongTensor(torch.zeros(
            (2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        c = torch.LongTensor(torch.zeros(
            (2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        if self.device:
            h, c = h.to(self.device), c.to(self.device)
        return h, c

    def forward(self, x_l, x_t):
        """

        :param x: (batch_size, seq_len), sequence of locations
        :return:
            (batch_size * seq_len, total_locations), prediction of next stage of all locations
        """
        locs = x_l.contiguous().view(-1).detach().cpu().numpy()
        mat1 = self.M1[locs]
        mat2 = self.M2[locs]
        mat1 = torch.Tensor(mat1).to(self.device)
        mat2 = torch.Tensor(mat2).to(self.device)


        lemb = self.loc_embedding(x_l)
        temb = self.tim_embedding(x_t)
        x = torch.cat([lemb, temb], dim=-1)
        
        x = x.transpose(0,1)
        Query = self.Q(x)
        Query = F.relu(Query)
       
        Value = self.V(x)
        Value = F.relu(Value)

        Key = self.K(x)
        Key = F.relu(Key)    

        x, _ = self.attn(Query, Key, Value)

        
        Query = self.Q2(x)
        Query = F.relu(Query)
       
        Value = self.V2(x)
        Value = F.relu(Value)

        Key = self.K2(x)
        Key = F.relu(Key)    

        x,_ = self.attn2(Query, Key, Value)       
        
        x = x.transpose(0,1)

        x = x.reshape(-1, self.linear_dim)
        x = self.linear(x)   
        x = F.relu(x)   

        
        mat1 = F.relu(self.linear_mat1(mat1))
        mat1 = torch.sigmoid(self.linear_mat1_2(mat1))
        mat1 = F.normalize(mat1)
        
        mat2 = F.relu(self.linear_mat2(mat2))
        mat2 = torch.sigmoid(self.linear_mat2_2(mat2))
        mat2 = F.normalize(mat2)

        

        pred = None

        if self.function:
            mat3 = self.M3[locs]
            mat3 = torch.Tensor(mat3).to(self.device)
            mat3 = torch.sigmoid(self.linear_mat3(mat3))
            pred = self.final_linear(x + torch.mul(x,mat1) + torch.mul(x,mat2) + torch.mul(x,mat3))
        else:
            pred = x + torch.mul(x,mat1) + torch.mul(x,mat2)
        pred = F.log_softmax(pred, dim=-1)


        return pred

    def step(self, l, t):
        """

        :param x: (batch_size, 1), current location
        :param h: (1/2, batch_size, hidden_dim), lstm hidden state
        :param c: (1/2, batch_size, hidden_dim), lstm cell state
        :return:
            (batch_size, total_locations), prediction of next stage
        """
        
        print(l.shape)
        #self.attn.flatten_parameters()
        locs = l.contiguous().view(-1).detach().cpu().numpy()
        mat1 = self.M1[locs]
        mat2 = self.M2[locs]
        mat1 = torch.Tensor(mat1).to(self.device)
        mat2 = torch.Tensor(mat2).to(self.device)


        lemb = self.loc_embedding(l)
        temb = self.tim_embedding(t)
        
        
        
        x = torch.cat([lemb, temb], dim=-1)
        
        x = x.transpose(0,1)

        Query = self.Q(x)
        Query = F.relu(Query)
       
        Value = self.V(x)
        Value = F.relu(Value)

        Key = self.K(x)
        Key = F.relu(Key)    

        x,_ = self.attn(Query, Key, Value)


        Query = self.Q2(x)
        Query = F.relu(Query)
       
        Value = self.V2(x)
        Value = F.relu(Value)

        Key = self.K2(x)
        Key = F.relu(Key)    

        x,_ = self.attn2(Query, Key, Value)       
        
        x = x.transpose(0,1)

        x = x.reshape(-1, self.linear_dim)
        x = self.linear(x)   
        x = F.relu(x)   

        
        mat1 = F.relu(self.linear_mat1(mat1))
        mat1 = torch.sigmoid(self.linear_mat1_2(mat1))
        mat1 = F.normalize(mat1)
        
        mat2 = F.relu(self.linear_mat2(mat2))
        mat2 = torch.sigmoid(self.linear_mat2_2(mat2))
        mat2 = F.normalize(mat2)

        

        pred = None

        if self.function:
            mat3 = self.M3[locs]
            mat3 = torch.Tensor(mat3).to(self.device)
            mat3 = torch.sigmoid(self.linear_mat3(mat3))
            pred = self.final_linear(x + torch.mul(x,mat1) + torch.mul(x,mat2) + torch.mul(x,mat3))
        else:
            pred = x + torch.mul(x,mat1) + torch.mul(x,mat2)
        pred = F.softmax(pred, dim=-1)


        return pred

    def sample(self, batch_size, seq_len, x=None):
        """

        :param batch_size: int, size of a batch of training data
        :param seq_len: int, length of the sequence
        :param x: (batch_size, k), current generated sequence
        :return: (batch_size, seq_len), complete generated sequence
        """
        res = []
        flag = False  # whether sample from zero
                
        #self.attn.flatten_parameters()

        if x is None:
            flag = True
        s = 0
        if flag:
            if self.starting_sample == 'zero':
                x = torch.LongTensor(torch.zeros((batch_size, 1))).to(self.device)
            elif self.starting_sample == 'rand':
                x  = torch.LongTensor(torch.randint(
                        high=self.total_locations, size=(batch_size, 1))).to(self.device)
            elif self.starting_sample == 'real':
                x = torch.LongTensor(torch.stack(
                    [torch.multinomial(self.starting_dist, 1) for i in range(batch_size)], dim=0)).to(self.device)
                s = 1

        if self.device:
            x = x.to(self.device)
        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                t = torch.LongTensor([i%24]).to(self.device)
                t = t.repeat(batch_size).reshape(batch_size, -1)
                x = self.step(x,t)
                x = x.multinomial(1)
                samples.append(x)
#                 print(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                t = torch.LongTensor([i%24]).to(self.device)           
                t= t.repeat(batch_size).reshape(batch_size, -1)
                x = self.step(lis[i],t)
                samples.append(lis[i])
            x = x.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)                                
                t = torch.LongTensor([i%24]).to(self.device)
                t = t.repeat(batch_size).reshape(batch_size, -1)
                x = self.step(x,t)
                x = x.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output 
