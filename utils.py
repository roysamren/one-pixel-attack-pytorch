'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - init_params: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
import numpy as np


def get_mean_and_std(dataset):
    '''Compute the mean and std value of a dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters (Conv2d, BatchNorm2d, Linear).'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')  # underscore version
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


# Attempt to get terminal width, fallback if it fails
try:
    rows, term_width_str = os.popen('stty size', 'r').read().split()
    term_width = int(term_width_str)
except:
    term_width = 80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    """
    A simple progress bar to mimic xlua.progress from Torch.
    """
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    sys.stdout.write('=' * cur_len)
    sys.stdout.write('>')
    sys.stdout.write('.' * rest_len)
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)
    msg_str = ''.join(L)
    sys.stdout.write(msg_str)

    # Fill the remaining width with spaces
    remaining = term_width - int(TOTAL_BAR_LENGTH) - len(msg_str) - 3
    if remaining > 0:
        sys.stdout.write(' ' * remaining)

    # Move back to center
    back_len = term_width - int(TOTAL_BAR_LENGTH / 2) + 2
    for _ in range(back_len):
        sys.stdout.write('\b')

    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds // 3600 // 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds // 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds // 60)
    secondsf = int(seconds - minutes * 60)
    millis = int((seconds - secondsf) * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



class DeepONetDataset(Dataset):
    ''' 
    Custom dataset class for the DeepONet model.
    
    Args:
    - branch_data: Branch input data, shape (num_samples, input_size)
    - trunk_data: Trunk input data, shape (num_trunk_points, trunk_size)
    - target_data: Target output data, shape (num_samples, num_trunk_points)
    
    This dataset assumes the trunk input is shared across all samples.
    '''
    
    def __init__(self, branch_data, trunk_data, target_data):
        self.branch_data = torch.tensor(branch_data, dtype=torch.float32)
        self.trunk_data = torch.tensor(trunk_data, dtype=torch.float32)    # Shared trunk input (100, 1)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.branch_data)  # Return the number of samples

    def __getitem__(self, idx):
        # Get the branch input and target output for this index
        branch_input = self.branch_data[idx]
        target_output = self.target_data[idx]
        # Return the branch input, shared trunk input (same for all samples), and target output
        return branch_input, self.trunk_data, target_output

class MIMONetDataset(Dataset):
    def __init__(self, branch_data_list, trunk_data, target_data):
        """
        Args:
            branch_data_list (list of np.ndarray): List of branch data arrays.
            trunk (np.ndarray): Trunk data array.
            target (np.ndarray): Target data array.
        """
        # Convert each branch input to a PyTorch tensor
        self.branches = [torch.tensor(branch, dtype=torch.float32) for branch in branch_data_list]
        self.trunk = torch.tensor(trunk_data, dtype=torch.float32)
        self.target = torch.tensor(target_data, dtype=torch.float32)

    def __len__(self):
        return len(self.target)  # Assuming target length matches the dataset size

    def __getitem__(self, idx):
        # For each sample, return a tuple of all branch inputs, the trunk input, and the target output
        branch_inputs = [branch[idx] for branch in self.branches]
        target_output = self.target[idx]

        # Return as tuple: list of branch inputs, trunk input, target output
        return branch_inputs, self.trunk, target_output

# scaler for target dataimport numpy as np
class ChannelScaler:
    def __init__(self, method='standard', feature_range=(-1, 1)):
        assert method in ['standard', 'minmax'], "method must be 'standard' or 'minmax'"
        self.method = method
        self.feature_range = feature_range
        self.params = []  # to hold (mean, std) or (min, max) per channel

    def fit(self, data):
        """ Fit scaler on training data only.
            data: numpy array with shape [samples, gridpoints, channels]
        """
        self.params = []
        for i in range(data.shape[2]):
            channel_data = data[:, :, i]
            if self.method == 'standard':
                mean = np.mean(channel_data)
                std = np.std(channel_data) + 1e-8
                self.params.append((mean, std))
            elif self.method == 'minmax':
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                self.params.append((min_val, max_val))
    
    def transform(self, data):
        """ Apply the scaling to input data using fitted stats. """
        scaled = np.empty_like(data)
        for i in range(data.shape[2]):
            x = data[:, :, i]
            if self.method == 'standard':
                mean, std = self.params[i]
                scaled[:, :, i] = (x - mean) / std
            elif self.method == 'minmax':
                min_val, max_val = self.params[i]
                scale = max_val - min_val + 1e-8
                a, b = self.feature_range
                scaled[:, :, i] = (b - a) * (x - min_val) / scale + a
        return scaled

    def inverse_transform(self, data):
        """ Recover original values from scaled data. """
        recovered = np.empty_like(data)
        for i in range(data.shape[2]):
            x = data[:, :, i]
            if self.method == 'standard':
                mean, std = self.params[i]
                recovered[:, :, i] = x * std + mean
            elif self.method == 'minmax':
                min_val, max_val = self.params[i]
                a, b = self.feature_range
                scale = (max_val - min_val + 1e-8)
                recovered[:, :, i] = ((x - a) * scale / (b - a)) + min_val
        return recovered