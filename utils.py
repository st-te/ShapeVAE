import random
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split

device = torch.device("cuda")

def set_seed(seed):
    """Sets seeds and worker initialization function for reproducibility."""
    torch.set_default_dtype(torch.float32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id):
    """Initialize each worker with unique seed derived from torch's initial seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def prepare_indices(data, base_dir, test_size=0.1, val_size=0.2, random_state=42):
    """
    Splits data into train, validation, and test indices and saves them as .npy files.
    """
    os.makedirs(base_dir, exist_ok=True)
    data_indices = list(range(len(data)))
    train_val_indices, test_indices = train_test_split(data_indices, test_size=test_size, random_state=random_state)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=val_size, random_state=random_state)

    # Save indices
    #np.save(os.path.join(base_dir, 'train_indices_706.npy'), train_indices)
    #np.save(os.path.join(base_dir, 'val_indices_706.npy'), val_indices)
    #np.save(os.path.join(base_dir, 'test_indices_706.npy'), test_indices)

    return train_indices, val_indices, test_indices

def compute_mape_overall(true, pred):
    """
    Compute MAPE (Mean Absolute Percentage Error).
    """
    if isinstance(true, np.ndarray):
        true = torch.tensor(true, dtype=torch.float32, device=device)  # Convert to tensor on CUDA
    if isinstance(pred, np.ndarray):
        pred = torch.tensor(pred, dtype=torch.float32, device=device)

    true, pred = true.float().to(device), pred.float().to(device)  # Ensure both are float32 and on CUDA
    return torch.mean(torch.abs((true - pred) / true)) * 100

def compute_mape_range(y_true, y_pred, lower_threshold=0.75, upper_threshold=1.25):
    """
    Compute MAPE for different value ranges.

    Args:
        y_true: Ground truth values (numpy array or torch tensor)
        y_pred: Predicted values (numpy array or torch tensor)
        lower_threshold: Threshold for low range values
        upper_threshold: Threshold for high range values

    Returns:
        dict: MAPE values for low, mid, and high ranges
    """
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.tensor(y_pred, dtype=torch.float32, device=device)

    y_true, y_pred = y_true.float().to(device), y_pred.float().to(device)
    epsilon = torch.finfo(torch.float32).eps

    mask_low = y_true < lower_threshold
    mask_mid = (y_true >= lower_threshold) & (y_true < upper_threshold)
    mask_high = y_true >= upper_threshold

    def calculate_mape(mask):
        if mask.sum().item() == 0:
            return 0.0
        values_true = y_true[mask]
        values_pred = y_pred[mask]
        denominator = torch.maximum(torch.abs(values_true), torch.tensor(epsilon, device=device))
        return (torch.mean(torch.abs((values_true - values_pred) / denominator)) * 100.0).item()

    return {
        'mape_low': calculate_mape(mask_low),
        'mape_mid': calculate_mape(mask_mid),
        'mape_high': calculate_mape(mask_high)
    }
