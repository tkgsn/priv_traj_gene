# coding: utf-8
import pdb
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from my_utils import get_gps


class GANLoss(nn.Module):
    """Reward-Refined NLLLoss Function for adversial training of Gnerator"""

    def __init__(self, cuda_number):
        super(GANLoss, self).__init__()
        self.cuda_number = cuda_number

#     def forward(self, prob, target, reward, ploss=False):
#         loss_model = nn.NLLLoss(reduction="none")
#         N = target.size(0)
#         C = prob.shape[1]
#         loss = loss_model(prob, target.to(prob.device))
#         loss = loss * reward
#         loss = torch.mean(loss)
#         return loss

    def forward(self, prob, target, reward, ploss=False):
        N = target.size(0)
        C = prob.shape[1]
        one_hot = torch.zeros((N, C))
        one_hot.scatter_(1, target.data.view((-1, 1)), 1)
        one_hot = one_hot.type(torch.bool)
        one_hot = Variable(one_hot)
        one_hot = one_hot.cuda(self.cuda_number)
        loss = torch.masked_select(torch.exp(prob), one_hot)
        loss = loss * reward
        loss = torch.mean(loss)
        return loss

class distance_loss(nn.Module):

    def __init__(self, dataset, M2, device, window_size):
        super(distance_loss, self).__init__()

        self.X, self.Y = get_gps(dataset)
        self.X = torch.Tensor(np.array(self.X)).float()
        self.Y = torch.Tensor(np.array(self.Y)).float()
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.n_vocabs = len(dataset.vocab)
        self.n_locations = dataset.n_locations
        self.seq_len = dataset.seq_len
        self.M2 = M2
        self.device = device
        self.window_size = window_size
        
        # print(self.X)

    def forward(self, prob, samples, start_time):
        """

        :param x: generated sequence, batch_size * seq_len
        :return:
        """
        batch_size = len(samples)
        samples = samples[:,start_time:start_time+self.window_size].reshape(-1).cpu().detach().numpy()
        distances = self.M2[samples]
        
        distances = torch.tensor(distances).to(self.device)
        loss = torch.exp(prob) * distances
        loss = torch.sum(loss) / (batch_size)
            
        return loss

def compute_distances(locations, target_location, M2):
    n_locations = M2.shape[0]
    
    return [M2[target_location, v] if (v < n_locations) and (target_location < n_locations) else 0 for v in locations]
    
def compute_kronegger_distances(locations, target_location, M2):
    n_locations = M2.shape[0]
    
    return [int(not (v == target_location)) if (v < n_locations) and (target_location < n_locations) else 0 for v in locations]