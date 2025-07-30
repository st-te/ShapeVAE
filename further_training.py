import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import r2_score
import logging
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset_6 import CrystalDataset
from model import ShapeVAE
from losses import ShapeLoss, angles_table
from utils import set_seed, worker_init_fn, prepare_indices, compute_mape_overall, compute_mape_range

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_num_threads(12)
device = torch.device("cuda")

# Best hyperparam from x-validation
LATENT_SIZE = 32
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
NUM_EPOCHS = 300

def train_model(model, device, train_loader, optimizer, scheduler, loss_function, epoch):
    model.train()
    total_train_loss = 0
    all_preds, all_trues = [], []
    all_pred_azimuth, all_true_azimuth = [], []
    all_pred_elevation, all_true_elevation = [], []

    for images, true_std, azimuth_true, elevation_true in train_loader:
        images = images.to(device).float()
        true_std = true_std.to(device).float()
        azimuth_true = azimuth_true.to(device).float()
        elevation_true = elevation_true.to(device).float()

        optimizer.zero_grad()
        x_reconstructed, azimuth_pred, elevation_pred, mu, log_var = model(images)
        loss = loss_function(x_reconstructed, true_std, mu, log_var,
                           azimuth_pred, elevation_pred, azimuth_true, elevation_true)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()

        all_preds.extend(x_reconstructed.detach().cpu().numpy())
        all_trues.extend(true_std.detach().cpu().numpy())
        all_pred_azimuth.extend(azimuth_pred.detach().cpu().numpy())
        all_true_azimuth.extend(azimuth_true.detach().cpu().numpy())
        all_pred_elevation.extend(elevation_pred.detach().cpu().numpy())
        all_true_elevation.extend(elevation_true.detach().cpu().numpy())

    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Epoch {epoch+1}: Current LR = {current_lr:.6f}")

    all_preds = np.vstack(all_preds).astype(np.float32)
    all_trues = np.vstack(all_trues).astype(np.float32)
    all_pred_azimuth = np.array(all_pred_azimuth, dtype=np.float32)
    all_true_azimuth = np.array(all_true_azimuth, dtype=np.float32)
    all_pred_elevation = np.array(all_pred_elevation, dtype=np.float32)
    all_true_elevation = np.array(all_true_elevation, dtype=np.float32)

    # metrics
    mape_train = compute_mape_overall(all_trues, all_preds)
    mape_details = compute_mape_range(all_trues, all_preds)
    r2_train = r2_score(all_trues, all_preds)
    r2_azimuth = r2_score(all_true_azimuth, all_pred_azimuth)
    r2_elevation = r2_score(all_true_elevation, all_pred_elevation)

    avg_train_loss = total_train_loss / len(train_loader)

    return avg_train_loss, mape_train, mape_details['mape_low'], mape_details['mape_mid'], \
           mape_details['mape_high'], r2_train, r2_azimuth, r2_elevation

def evaluate_model(model, device, loader, loss_function):
    model.eval()
    total_loss = 0
    all_preds, all_trues = [], []
    all_pred_azimuth, all_true_azimuth = [], []
    all_pred_elevation, all_true_elevation = [], []

    with torch.no_grad():
        for images, true_std, azimuth_true, elevation_true in loader:
            images = images.to(device).float()
            true_std = true_std.to(device).float()
            azimuth_true = azimuth_true.to(device).float()
            elevation_true = elevation_true.to(device).float()

            x_reconstructed, azimuth_pred, elevation_pred, mu, log_var = model(images)
            loss = loss_function(x_reconstructed, true_std, mu, log_var,
                               azimuth_pred, elevation_pred, azimuth_true, elevation_true)
            total_loss += loss.item()

            all_preds.extend(x_reconstructed.cpu().numpy())
            all_trues.extend(true_std.cpu().numpy())
            all_pred_azimuth.extend(azimuth_pred.cpu().numpy())
            all_true_azimuth.extend(azimuth_true.cpu().numpy())
            all_pred_elevation.extend(elevation_pred.cpu().numpy())
            all_true_elevation.extend(elevation_true.cpu().numpy())

    all_preds = np.vstack(all_preds).astype(np.float32)
    all_trues = np.vstack(all_trues).astype(np.float32)
    all_pred_azimuth = np.array(all_pred_azimuth, dtype=np.float32)
    all_true_azimuth = np.array(all_true_azimuth, dtype=np.float32)
    all_pred_elevation = np.array(all_pred_elevation, dtype=np.float32)
    all_true_elevation = np.array(all_true_elevation, dtype=np.float32)

    # metrics
    mape = compute_mape_overall(all_trues, all_preds)
    mape_details = compute_mape_range(all_trues, all_preds)
    r2_overall = r2_score(all_trues, all_preds)
    r2_azimuth = r2_score(all_true_azimuth, all_pred_azimuth)
    r2_elevation = r2_score(all_true_elevation, all_pred_elevation)
    avg_loss = total_loss / len(loader)

    return avg_loss, mape, mape_details['mape_low'], mape_details['mape_mid'], \
           mape_details['mape_high'], r2_overall, r2_azimuth, r2_elevation

def further_train():
    set_seed(42)
    device = torch.device("cuda")

    image_dir = os.getenv("IMAGE_DIR")
    std_dir = os.getenv("STD_DIR")
    indices_dir = os.getenv('INDICES')  # Use INDICES for everything
    results_dir = os.path.join(indices_dir, "further_training_results")
    os.makedirs(results_dir, exist_ok=True)

    from losses import angles_table

    data = CrystalDataset(image_dir, std_dir, angles_table)

    train_indices_dir = os.getenv('TRAIN_INDICES')
    train_indices = np.load(train_indices_dir)

    val_indices_dir = os.getenv('VAL_INDICES')
    val_indices = np.load(val_indices_dir)

    test_indices_dir = os.getenv('TEST_INDICES')
    test_indices = np.load(test_indices_dir)

    loaders = {
        'train': DataLoader(
            Subset(data, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            num_workers=0
        ),
        'val': DataLoader(
            Subset(data, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            num_workers=0
        ),
        'test': DataLoader(
            Subset(data, test_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            num_workers=0
        )
    }

    model = ShapeVAE().to(device)

    checkpoint_path = "/mnt/eps01-rds/Mike_Anderson_Group/Steven/VAE_706point_split_32/vae_model_lr0.0001_bs256_fold2.model"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Loaded pretrained weights from fold 3 (0-indexed) with best azimuth R² performance")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}, starting with fresh weights")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    loss_function = ShapeLoss()

    # training history
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mape': [], 'val_mape': [], 'test_mape': [],
        'train_r2': [], 'val_r2': [], 'test_r2': [],
        'train_r2_azimuth': [], 'val_r2_azimuth': [], 'test_r2_azimuth': [],
        'train_r2_elevation': [], 'val_r2_elevation': [], 'test_r2_elevation': [],
        'lr': []
    }

    best_val_r2 = -float('inf')

    # training loop
    for epoch in range(NUM_EPOCHS):

        train_metrics = train_model(model, device, loaders['train'], optimizer, scheduler, loss_function, epoch)
        val_metrics = evaluate_model(model, device, loaders['val'], loss_function)
        test_metrics = evaluate_model(model, device, loaders['test'], loss_function)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics[0])
        history['val_loss'].append(val_metrics[0])
        history['test_loss'].append(test_metrics[0])
        history['train_mape'].append(train_metrics[1].item())
        history['val_mape'].append(val_metrics[1].item())
        history['test_mape'].append(test_metrics[1].item())
        history['train_r2'].append(train_metrics[5])
        history['val_r2'].append(val_metrics[5])
        history['test_r2'].append(test_metrics[5])
        history['train_r2_azimuth'].append(train_metrics[6])
        history['val_r2_azimuth'].append(val_metrics[6])
        history['test_r2_azimuth'].append(test_metrics[6])
        history['train_r2_elevation'].append(train_metrics[7])
        history['val_r2_elevation'].append(val_metrics[7])
        history['test_r2_elevation'].append(test_metrics[7])
        history['lr'].append(optimizer.param_groups[0]['lr'])

        logger.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - "
                  f"Train Loss: {train_metrics[0]:.4f}, Val Loss: {val_metrics[0]:.4f}, Test Loss: {test_metrics[0]:.4f}, "
                  f"Train R2: {train_metrics[5]:.4f}, Val R2: {val_metrics[5]:.4f}, Test R2: {test_metrics[5]:.4f}")

        # save model if validation R2 improved
        if val_metrics[5] > best_val_r2:
            best_val_r2 = val_metrics[5]
            torch.save(model.state_dict(), os.path.join(results_dir, f"best_model_latent{LATENT_SIZE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.model"))
            logger.info(f"Saved new best model with validation R2: {best_val_r2:.4f}")

        pd.DataFrame(history).to_csv(os.path.join(results_dir, f"training_history_latent{LATENT_SIZE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.csv"), index=False)

    # save final model
    torch.save(model.state_dict(), os.path.join(results_dir, f"final_model_latent{LATENT_SIZE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.model"))

    logger.info("Final model evaluation:")
    final_train = evaluate_model(model, device, loaders['train'], loss_function)
    final_val = evaluate_model(model, device, loaders['val'], loss_function)
    final_test = evaluate_model(model, device, loaders['test'], loss_function)

    final_results = [{
        'latent_size': LATENT_SIZE,
        'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'epochs': NUM_EPOCHS,
        # training metrics
        'train_loss': final_train[0],
        'train_mape': final_train[1].item(),
        'train_mape_low': final_train[2],
        'train_mape_mid': final_train[3],
        'train_mape_high': final_train[4],
        'train_r2': final_train[5],
        'train_r2_azimuth': final_train[6],
        'train_r2_elevation': final_train[7],
        # validation metrics
        'val_loss': final_val[0],
        'val_mape': final_val[1].item(),
        'val_mape_low': final_val[2],
        'val_mape_mid': final_val[3],
        'val_mape_high': final_val[4],
        'val_r2': final_val[5],
        'val_r2_azimuth': final_val[6],
        'val_r2_elevation': final_val[7],
        # test metrics
        'test_loss': final_test[0],
        'test_mape': final_test[1].item(),
        'test_mape_low': final_test[2],
        'test_mape_mid': final_test[3],
        'test_mape_high': final_test[4],
        'test_r2': final_test[5],
        'test_r2_azimuth': final_test[6],
        'test_r2_elevation': final_test[7]
    }]

    pd.DataFrame(final_results).to_csv(os.path.join(results_dir, f"final_results_latent{LATENT_SIZE}_lr{LEARNING_RATE}_bs{BATCH_SIZE}.csv"), index=False)
    logger.info("Final results saved.")
    return final_results




if __name__ == "__main__":
    set_seed(42)
    image_dir = os.getenv("IMAGE_DIR")
    std_dir = os.getenv("STD_DIR")
    from losses import angles_table
    data = CrystalDataset(image_dir, std_dir, angles_table)
    further_train()
