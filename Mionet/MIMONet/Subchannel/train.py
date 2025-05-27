# %%
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import random
import matplotlib.pyplot as plt
#from scripts.training import train_model

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

# importing custom modules
from utils import MIMONetDataset, DeepONetDataset, ChannelScaler
from mimonet import MIMONet

from training import train_model

from train_utils import test_kfold_model, test_model

# %%
# check if GPU is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# %%
# set working directory
working_dir = "/projects/bcnx/kazumak2/MIMONet/Subchannel/"
data_dir = os.path.join(working_dir, "data")


# set random seed for reproducibility
seed = 12345
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# %% [markdown]
# ## load datasets

# %% [markdown]
# ### Load sharing parameters/dataset

# %%
# trunk dataset
trunk_input = np.load(os.path.join(data_dir, "share/trunk_input.npz"))['trunk']

# %% [markdown]
# ### Training data

# %%
# training data
train_branch = np.load(os.path.join(data_dir, "training/train_branch_input.npz"))
train_branch_1 = train_branch['func_params']
train_branch_2 = train_branch['stat_params']

# [samples, channel, gridpoints]
train_target = np.load(os.path.join(data_dir, "training/train_target.npz"))['target']
# convert to [samples, gridpoints, channel]
train_target = np.moveaxis(train_target, 1, 2)

print("train_branch_1 shape:", train_branch_1.shape)
print("train_branch_2 shape:", train_branch_2.shape)
print("train_target shape:", train_target.shape)

# %%
# scaling the functional input data using predefined mean and std
f_mean = np.load(os.path.join(data_dir, "share/func_mean_std_params.npz"))['mean']
f_std = np.load(os.path.join(data_dir, "share/func_mean_std_params.npz"))['std']

train_branch_1 = (train_branch_1 - f_mean) / f_std

# scaling the static input data using predefined mean and std
s_mean = np.load(os.path.join(data_dir, "share/stat_mean_std_params.npz"))['mean']
s_std = np.load(os.path.join(data_dir, "share/stat_mean_std_params.npz"))['std']

for i in range(s_mean.shape[0]):
    train_branch_2[:, i] = (train_branch_2[:, i] - s_mean[i]) / s_std[i]

# %% [markdown]
# ### Test data

# %%
test_branch = np.load(os.path.join(data_dir, "test/test_branch_input.npz"))
test_branch_1 = test_branch['func_params']
test_branch_2 = test_branch['stat_params']

test_target = np.load(os.path.join(data_dir, "test/test_target.npz"))['target']
test_target = np.moveaxis(test_target, 1, 2)

print("test_branch_1 shape:", test_branch_1.shape)
print("test_branch_2 shape:", test_branch_2.shape)
print("test_target shape:", test_target.shape)

# scaling the functional input data using predefined mean and std
test_branch_1 = (test_branch_1 - f_mean) / f_std
# scaling the static input data using predefined mean and std
for i in range(s_mean.shape[0]):
    test_branch_2[:, i] = (test_branch_2[:, i] - s_mean[i]) / s_std[i]

# %% [markdown]
# ### Scaling the target data

# %%
# scaling the target data
'''  
note: reverse the scaling for the target data
train_target = scaler.inverse_transform(train_target_scaled)
test_target = scaler.inverse_transform(test_target_scaled)
'''
scaler = ChannelScaler(method='minmax', feature_range=(-1, 1))
scaler.fit(train_target)
train_target_scaled = scaler.transform(train_target)
test_target_scaled = scaler.transform(test_target)


# %% [markdown]
# ## Torch Dataset and DataLoader

# %%
# test dataset and dataloader
test_dataset = MIMONetDataset(
    [test_branch_1, test_branch_2],  # branch_data_list
    trunk_input,                     # trunk_data
    test_target_scaled               # target_data
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,  # set to 1 for testing
    shuffle=False,
    num_workers=0
)

# %%
train_dataset = MIMONetDataset(
    [train_branch_1, train_branch_2],  # branch_data_list
    trunk_input,                       # trunk_data
    train_target_scaled                # target_data
)

# %% [markdown]
# ## MIMONet Model

# %%
# Architecture parameters
dim = 256
branch_input_dim1 = 100
branch_input_dim2 = 2
trunk_input_dim = 2


model_args = {
    'branch_arch_list': [
        [branch_input_dim1, 512, 512, 512, dim],
        [branch_input_dim2, 512, 512, 512, dim]
    ],
    'trunk_arch': [trunk_input_dim, 256, 256, 256, dim],
    'num_outputs': 3,
    'activation_fn': nn.ReLU,
    'merge_type': 'mul'
}

# scheduler parameters
scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': 10,}

# model training script
train_model(
    model_fn=MIMONet,
    model_args=model_args,
    optimizer_fn=torch.optim.Adam,
    optimizer_args={'lr': 1e-3, 'weight_decay': 1e-6},
    scheduler_fn=scheduler_fn,
    scheduler_args=scheduler_args,
    dataset=train_dataset,
    device=device,
    num_epochs=500,
    batch_size=4,
    criterion=nn.MSELoss(),
    patience=500,
    k_fold=None,
    multi_gpu=False,
    working_dir=""
)

print("Training completed.")

## Evaluation
#train_mode = 'k_fold'
train_mode = 'default'
n_hold = 5

# initialize the model using model_args
model = MIMONet(**model_args).to(device)

if train_mode == 'k_fold':
    for i in range(n_hold):
        best_model_path = os.path.join(working_dir, f"checkpoints/best_model_fold{i+1}.pt")
        
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.to(device)
            model.eval()
            print(f"Best model for fold {i+1} loaded.")
        else:
            print(f"Best model for fold {i+1} not found. Please check the path.")
            exit(1)
        
        test_kfold_model(
            fold_id=i+1,
            model=model,
            test_loader=test_loader,
            scaler=scaler,
            working_dir=working_dir,
            device=device,
            test_branch=test_branch,
            save_array=True
        )   
    
else:
    # Load the best model (best_model.pt)
    best_model_path = os.path.join(working_dir, "checkpoints/best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        model.to(device)
        model.eval()
        print("Best model loaded.")
    else:
        print("Best model not found. Please check the path.")
        exit(1)
    
    # Test the model
    test_model(
        model=model,
        test_loader=test_loader,
        scaler=scaler,
        working_dir=working_dir,
        device=device,
        test_branch=test_branch,
        save_array=True
        )