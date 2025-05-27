import torch
from torch.utils.data import Dataset
import numpy as np


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
