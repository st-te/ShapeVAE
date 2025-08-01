# ShapeVAE

A PyTorch implementation of a Variational Autoencoder (VAE) for reconstructing 3D crystal morphologies from 2D image data and predicting viewing angles.

## Overview

This project trains a VAE to:
- Reconstruct 3D crystal morphologies (706-dimensional tensor representation of unit sphere scalars) from 2D images (120x120 grayscale)
- Predict viewing angles (azimuth: 0-360°, elevation: 0-180°) 
- Learn disentangled latent representations separating shape and orientation information

## Architecture

**Encoder**: CNN-based encoder that processes 2D images and outputs separate latent representations for:
- Orientation features (2D) → azimuth/elevation prediction
- Shape features (30D for latent_dim=32) → 3D structure reconstruction

**Decoder**: Two-headed decoder with:
- Orientation prediction head → azimuth and elevation angles
- Shape reconstruction head → 706-dimensional representation of 3D structure

## Project Structure

```
├── model.py           # VAE architecture (Encoder, CNNDecoder, ShapeVAE)
├── losses.py          # Combined loss function (reconstruction + angles + KL)
├── dataset_6.py       # Dataset class for crystal images and 3D data
├── utils.py           # Utilities (seeding, metrics, data splitting)
├── further_training.py # Main training script with pretrained model loading
└── README.md          # This file
```

## Requirements

```bash
torch
torchvision
numpy
scikit-learn
pandas
PIL
```

## Environment Variables

Set the following environment variables before running:

```bash
export IMAGE_DIR="/path/to/crystal/images"      # Directory containing simulation subdirs with PNG files
export STD_DIR="/path/to/standard/data"        # Directory with .npy files (706-dimensional targets)
export INDICES="/path/to/indices"              # Base directory for train/val/test splits
export TRAIN_INDICES="/path/to/train_indices.npy"
export VAL_INDICES="/path/to/val_indices.npy"
export TEST_INDICES="/path/to/test_indices.npy"
```

## Data Format

**Images**: 120x120 grayscale PNG files organised as:
```
IMAGE_DIR/
├── simulation_001/
│   ├── sim_001_0.png    # angle_idx=0 → (azimuth=0°, elevation=0°)
│   ├── sim_001_1.png    # angle_idx=1 → (azimuth=0°, elevation=63.43°)
│   └── ...
└── simulation_002/
    └── ...
```

**3D Data**: NumPy arrays (706-dimensional) in:
```
STD_DIR/
├── simulation_001/
│   └── simulation_001.npy
└── simulation_002/
    └── simulation_002.npy
```

## Usage

### Training

```bash
python further_training.py
```

**Key Parameters**:
Parameters used in the final training from k-fold cross-validation:
- `LATENT_SIZE = 32` (total latent dimensions: 2 for orientation + 30 for shape)
- `LEARNING_RATE = 0.0001`
- `BATCH_SIZE = 256`
- `NUM_EPOCHS = 300`

### Pretrained Model

The script loads a pretrained model.

Update the `checkpoint_path` variable to use your pretrained model location.

## Outputs

Training produces:
- `best_model_*.model` - Best model based on validation R²
- `final_model_*.model` - Final model after all epochs
- `training_history_*.csv` - Loss and metrics per epoch
- `final_results_*.csv` - Final evaluation metrics

## Evaluation Metrics

- **R² Score**: Overall reconstruction quality and angle prediction accuracy
- **MAPE**: Mean Absolute Percentage Error for 3D reconstruction
  - Overall MAPE
  - Range-specific MAPE (low: <0.75, mid: 0.75-1.25, high: >1.25)
- **MSE**: Mean Squared Error for angle predictions (with periodic handling for azimuth)

## Loss Function

Combined loss with configurable weights:
```python
total_loss = reconstruction_loss + λ_angles * angle_loss + λ_kl * kl_divergence
```

Where:
- `λ_angles = 0.25` (azimuth/elevation loss weight)
- `λ_kl = 0.25` (KL divergence weight)

## Key Features

- **Angle Table**: Predefined viewing angles for 6 orientations (0-5)
- **Gradient Clipping**: Max norm of 1.0 to prevent exploding gradients
- **Learning Rate Scheduling**: Cosine annealing with minimum LR of 1e-6
- **Reproducibility**: Fixed seeds and deterministic operations
- **GPU Support**: CUDA-optimized with mixed precision support

## Notes

- Input images are normalised to [-1, 1] range
- 3D outputs are constrained to [0, 2] range via Hardtanh
- Azimuth predictions handle periodic boundary conditions (0° = 360°)
- Model uses LeakyReLU activations and batch normalisation throughout
