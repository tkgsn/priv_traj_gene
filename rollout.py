# -*- coding:utf-8 -*-

import copy
import numpy as np
import torch

class Rollout(object):
    """Roll-out policy"""
    def __init__(self, model, update_rate, cuda_number=0):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.cuda_number = cuda_number
        
    def get_lazy_reward(self, x, num, discriminator, window_size):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        pred = discriminator(x.cuda(self.cuda_number))
        pred = pred.detach().cpu().data[:,1].numpy()
        pred = np.concatenate([[pred[i]]*window_size for i in range(batch_size)])

        rewards = np.transpose(np.array(pred)) / (1.0 * window_size) # batch_size * seq_len
        return rewards


    def get_reward(self, x, num, discriminator):
        
        rewards = []
        batch_size = x.size(0)
        seq_len = x.size(1)
        start_indice = torch.tensor([[self.own_model.start_index]]*batch_size)
        for i in range(num):
            for l in range(1, seq_len+1):
                data = x.contiguous()
                data[:, l:] = self.own_model.mask_index
                data = torch.concat([start_indice, data], dim=1)
                samples = self.own_model.sample(batch_size, l, data).cuda(self.cuda_number)
                pred = discriminator(samples)
                pred = pred.detach().cpu().data[:,1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l-1] += pred

        rewards = np.transpose(np.array(rewards)) / (1.0 * num) # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith('emb') or name.startswith('Emb'):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
#                 param.data = dic[name]
