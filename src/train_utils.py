import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import KFold
import numpy as np


def test_kfold_model(fold_id, model, test_loader, scaler, working_dir, device, test_branch, save_array=False):
    predictions = []
    targets = []

    with torch.no_grad():
        for (branch_data, trunk_data, target_data) in test_loader:
            branch_data = [b.to(device) for b in branch_data]
            trunk_data = trunk_data.to(device)
            target_data = target_data.to(device)

            output = model(branch_data, trunk_data)

            predictions.append(output.cpu().numpy())
            targets.append(target_data.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    predictions = scaler.inverse_transform(predictions)
    targets = scaler.inverse_transform(targets)

    if save_array == False:
        # pass nothing to do
        pass
    elif save_array == True:
        # Save the predictions and targets in same npz file
        output_path = os.path.join(working_dir, f"results/test_results_fold_{fold_id}.npz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.savez(
            output_path,
            branch1=test_branch['func_params'],
            branch2=test_branch['stat_params'],
            predictions=predictions,
            targets=targets
        )
        print(f"Saved test results for fold {fold_id} to {output_path}")
    
    # compute relative l2 errors for each channel (1000, 1733, 3)
    relative_l2_errors = np.zeros((predictions.shape[0], predictions.shape[2]))
    for i in range(predictions.shape[2]):
        relative_l2_errors[:, i] = np.linalg.norm(predictions[:, :, i] - targets[:, :, i], ord=2, axis=1) / np.linalg.norm(targets[:, :, i], ord=2, axis=1)
        
    # compute the mean and standard deviation of the relative l2 errors for each channel
    mean_errors = np.mean(relative_l2_errors, axis=0)
    std_errors = np.std(relative_l2_errors, axis=0)
    print("Mean relative L2 errors:", mean_errors)
    print("Standard deviation of relative L2 errors:", std_errors)


def test_model(model, test_loader, scaler, working_dir, device, test_branch, save_array=False):
    predictions = []
    targets = []
    with torch.no_grad():
        for i, (branch_data, trunk_data, target_data) in enumerate(test_loader):
            branch_data = [data.to(device) for data in branch_data]
            trunk_data = trunk_data.to(device)
            target_data = target_data.to(device)

            # Forward pass
            output = model(branch_data, trunk_data)

            # Store predictions and targets
            predictions.append(output.cpu().numpy())
            targets.append(target_data.cpu().numpy())

    # Convert predictions and targets to numpy arrays
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    # Inverse transform the predictions and targets
    predictions = scaler.inverse_transform(predictions)
    targets = scaler.inverse_transform(targets)

    if save_array == False:
        pass
    elif save_array == True:
        # Save the predictions and targets in same npz file
        output_path = os.path.join(working_dir, "results/test_results.npz")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        np.savez(
            output_path,
            branch1=test_branch['func_params'],
            branch2=test_branch['stat_params'],
            predictions=predictions,
            targets=targets
        )
        print(f"Saved test results to {output_path}")

    # compute relative l2 errors for each channel (1000, 1733, 3)
    relative_l2_errors = np.zeros((predictions.shape[0], predictions.shape[2]))
    for i in range(predictions.shape[2]):
        relative_l2_errors[:, i] = np.linalg.norm(predictions[:, :, i] - targets[:, :, i], ord=2, axis=1) / np.linalg.norm(targets[:, :, i], ord=2, axis=1)
        
    # compute the mean and standard deviation of the relative l2 errors for each channel
    mean_errors = np.mean(relative_l2_errors, axis=0)
    std_errors = np.std(relative_l2_errors, axis=0)
    print("Mean relative L2 errors:", mean_errors)
    print("Standard deviation of relative L2 errors:", std_errors)