import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from torch.utils.data import Dataset
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int = 15,
                 cycle_mult : float = 2,
                 max_lr : float = 0.1,
                 min_lr : float = 1e-6,
                 warmup_steps : int = 5,
                 gamma : float = 0.9,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class SVC_Dataset(Dataset):
    def __init__(self, data_ori, cell_morphology_vec, nuclear_morphology_vec, location, identity_vec=None):
        self.data_ori = data_ori
        self.cell_morphology_vec = cell_morphology_vec
        self.nuclear_morphology_vec = nuclear_morphology_vec
        self.location = location
        if identity_vec is not None:
            self.identity_vec = identity_vec
        
    def __len__(self):
        return len(self.data_ori)

    def __getitem__(self, index):
        if hasattr(self, 'identity_vec'):
            return  self.data_ori[index], self.cell_morphology_vec[index], self.nuclear_morphology_vec[index],  self.location[index], self.identity_vec[index]
        else:
            return  self.data_ori[index], self.cell_morphology_vec[index], self.nuclear_morphology_vec[index],  self.location[index]


def negative_binomial_loss(y_true, mu, r):
    y = y_true.float()
    term1 = torch.special.gammaln(y + r)
    term2 = torch.special.gammaln(y + 1)
    term3 = torch.special.gammaln(r)
    term4 = r * (torch.log(r) - torch.log(r + mu))
    term5 = y * (torch.log(mu) - torch.log(r + mu))
    
    log_likelihood = term1 - term2 - term3 + term4 + term5
    return -log_likelihood

def find_closest_point(df1, df2, angle_col, type_col):
    closest_points = []
    for _, row in df1.iterrows():
        angle_diffs = np.abs(df2[angle_col] - row[angle_col])
        idx = angle_diffs.argmin()
        df2_idx = df2.iloc[idx]
        if df2_idx.empty:
            print("empty")
            continue
        closest_points.append(df2_idx['distance_to_center']*row['ratio'])

    return closest_points


def create_white_to_color_cmap(target_color, name='custom_cmap'):
    colors = [
        (1, 1, 1),         
        to_rgba(target_color)  
    ]

    return LinearSegmentedColormap.from_list(name, colors)