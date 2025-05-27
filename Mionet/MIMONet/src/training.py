import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Subset, DistributedSampler
from sklearn.model_selection import KFold
import numpy as np

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    for branch_batch, trunk_batch, target_batch in dataloader:
        branch_batch = [b.to(device) for b in branch_batch]
        trunk_batch = trunk_batch.to(device)
        target_batch = target_batch.to(device)

        output = model(branch_batch, trunk_batch)
        loss = criterion(output.squeeze(-1), target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for branch_batch, trunk_batch, target_batch in dataloader:
            branch_batch = [b.to(device) for b in branch_batch]
            trunk_batch = trunk_batch.to(device)
            target_batch = target_batch.to(device)

            output = model(branch_batch, trunk_batch)
            loss = criterion(output.squeeze(-1), target_batch)
            val_loss += loss.item()
    return val_loss / len(dataloader)


def train_model(
    model_fn,
    model_args: dict,
    optimizer_fn,
    optimizer_args: dict,
    dataset,
    scheduler_fn=None,
    scheduler_args=None,
    device='cuda',
    num_epochs=500,
    batch_size=64,
    criterion=nn.MSELoss(),
    patience=10,
    val_ratio=0.2,
    k_fold=None,
    multi_gpu=False,
    working_dir="experiment"
):

    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def step_scheduler(scheduler, val_loss=None):
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    if k_fold:
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=12345)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            print(f"\n=== Fold {fold+1}/{k_fold} ===")

            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)

            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=batch_size)

            # Reinitialize model, optimizer, scheduler
            model = model_fn(**model_args).to(device)
            if multi_gpu and torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs")
                model = torch.nn.DataParallel(model)

            optimizer = optimizer_fn(model.parameters(), **optimizer_args)
            scheduler = scheduler_fn(optimizer, **scheduler_args) if scheduler_fn else None

            best_val_loss = float('inf')
            epochs_no_improve = 0
            save_path = os.path.join(checkpoint_dir, f"best_model_fold{fold+1}.pt")

            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                val_loss = evaluate(model, val_loader, criterion, device)

                step_scheduler(scheduler, val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}, LR: {current_lr:.2e}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print("Early stopping triggered.")
                        break

            fold_results.append(best_val_loss)

        print(f"\n=== K-Fold Cross-Validation Results ===")
        print(f"Fold Losses: {fold_results}")
        print(f"Mean = {np.mean(fold_results):.6f}, Std = {np.std(fold_results):.6f}")
        return fold_results

    else:
        # Standard train/val split
        val_len = int(len(dataset) * val_ratio)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)

        # Reinitialize model, optimizer, scheduler once
        model = model_fn(**model_args).to(device)
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = torch.nn.DataParallel(model)

        optimizer = optimizer_fn(model.parameters(), **optimizer_args)
        scheduler = scheduler_fn(optimizer, **scheduler_args) if scheduler_fn else None

        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_losses, val_losses = [], []
        save_path = os.path.join(checkpoint_dir, "best_model.pt")

        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            step_scheduler(scheduler, val_loss)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, LR: {current_lr:.2e}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered.")
                    break

        return train_losses, val_losses

##
def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def train_model_ddp(
    model_fn,
    model_args: dict,
    optimizer_fn,
    optimizer_args: dict,
    dataset,
    rank,
    world_size,
    scheduler_fn=None,
    scheduler_args=None,
    num_epochs=500,
    batch_size=64,
    criterion=nn.MSELoss(),
    patience=10,
    val_ratio=0.2,
    k_fold=None,
    working_dir="experiment"
):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    def step_scheduler(scheduler, val_loss=None):
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    if k_fold:
        kfold = KFold(n_splits=k_fold, shuffle=True, random_state=12345)
        fold_results = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
            train_subset = Subset(dataset, train_ids)
            val_subset = Subset(dataset, val_ids)

            train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank)

            train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(val_subset, batch_size=batch_size, sampler=val_sampler)

            model = model_fn(**model_args).to(device)
            model = DDP(model, device_ids=[rank], output_device=rank)

            optimizer = optimizer_fn(model.parameters(), **optimizer_args)
            scheduler = scheduler_fn(optimizer, **scheduler_args) if scheduler_fn else None

            best_val_loss = float('inf')
            epochs_no_improve = 0
            save_path = os.path.join(checkpoint_dir, f"best_model_fold{fold+1}_rank{rank}.pt")

            for epoch in range(num_epochs):
                model.train()
                train_sampler.set_epoch(epoch)
                total_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    loss = train_batch(batch, model, criterion, device)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        val_loss += evaluate_batch(batch, model, criterion, device)
                val_loss /= len(val_loader)
                step_scheduler(scheduler, val_loss)

                current_lr = optimizer.param_groups[0]['lr']
                if rank == 0:
                    print(f"[Fold {fold+1}] Epoch {epoch+1}, LR: {current_lr:.2e}, Train Loss: {total_loss:.6f}, Val Loss: {val_loss:.6f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    if rank == 0:
                        torch.save(model.state_dict(), save_path)
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if rank == 0:
                            print("Early stopping triggered.")
                        break

            fold_results.append(best_val_loss)

        if rank == 0:
            print(f"DDP K-Fold Results: {fold_results}, Mean = {np.mean(fold_results):.6f}, Std = {np.std(fold_results):.6f}")

    cleanup_ddp()

def train_batch(batch, model, criterion, device):
    branch_data, trunk_data, target_data = batch
    branch_data = [x.to(device) for x in branch_data]
    trunk_data = trunk_data.to(device)
    target_data = target_data.to(device)
    output = model(branch_data, trunk_data)
    loss = criterion(output, target_data)
    return loss

def evaluate_batch(batch, model, criterion, device):
    branch_data, trunk_data, target_data = batch
    branch_data = [x.to(device) for x in branch_data]
    trunk_data = trunk_data.to(device)
    target_data = target_data.to(device)
    output = model(branch_data, trunk_data)
    loss = criterion(output, target_data)
    return loss.item()

def run_ddp_training(model_fn, model_args, optimizer_fn, optimizer_args, dataset, world_size, **kwargs):
    mp.spawn(
        train_model_ddp,
        args=(model_fn, model_args, optimizer_fn, optimizer_args, dataset, world_size, *kwargs.values()),
        nprocs=world_size,
        join=True
    )
