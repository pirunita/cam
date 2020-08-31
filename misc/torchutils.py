import torch
from torch.utils.data import Subset

import numpy as np

class PolyOptimizer(torch.optim.SGD):
    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)
        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum
        
        self.__initial_lr = [group['lr'] for group in self.param_groups]
    
    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1
        

def split_dataset(dataset, n_splits):
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]

def save_checkpoint(state, filename):
    print('save')
    try:
        torch.save(state, filename)
    except:
        raise Exception

