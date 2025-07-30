import torch
import torch.nn.functional as F
import random
import numpy as np

device = torch.device("cuda")

angles_table = {
    0: (0, 0),
    1: (0, 63.43494882),
    2: (288, 63.43494882),
    3: (144, 63.43494882),
    4: (216, 63.43494882),
    5: (72, 63.43494882)
}

class ShapeLoss(torch.nn.Module):
    def __init__(self, lambda_az_el=0.25, lambda_kl=0.25):
        """
        Initialise the Shape Loss module with configurable weights.

        Args:
            lambda_az_el (float): Weight for azimuth/elevation loss component
            lambda_kl (float): Weight for KL divergence loss component
        """
        super(ShapeLoss, self).__init__()
        self.lambda_az_el = lambda_az_el
        self.lambda_kl = lambda_kl
        self.epsilon = torch.finfo(torch.float32).eps

    def compute_angle_loss(self, pred, true, scale):
        # Handle periodic nature of azimuth
        if scale == 360.0:  # azimuth case
            diff = (pred - true) % 360.0
            diff = torch.minimum(diff, 360.0 - diff)
            return (diff / scale).pow(2).mean()
        return F.mse_loss(pred/scale, true/scale)

    def forward(self, x_reconstructed, x, mu, log_var, azimuth_pred, elevation_pred,
                azimuth_true, elevation_true):
        """
        Compute the combined loss with improved numerical stability.

        Args:
            x_reconstructed: Reconstructed shape data
            x: Original shape data
            mu: Mean of the latent distribution
            log_var: Log variance of the latent distribution
            azimuth_pred: Predicted azimuth angles
            elevation_pred: Predicted elevation angles
            azimuth_true: Ground truth azimuth angles
            elevation_true: Ground truth elevation angles
        """
        device = x.device
        x_reconstructed = x_reconstructed.float().to(device)
        x = x.float().to(device)
        mu = mu.float().to(device)
        log_var = log_var.float().to(device)

        # 1. Reconstruction Loss (MSE)
        mse_loss = F.mse_loss(x_reconstructed, x, reduction='mean')

        # 2. Angle Losses with appropriate scaling
        mse_azimuth = self.compute_angle_loss(azimuth_pred, azimuth_true, 360.0)
        mse_elevation = self.compute_angle_loss(elevation_pred, elevation_true, 180.0)
        angle_loss = self.lambda_az_el * (mse_azimuth + mse_elevation)

        # 3. KL Divergence with numerical stability
        # Clamp values to prevent numerical issues
        log_var = torch.clamp(log_var, min=-20, max=20)  # Prevent extreme values
        var = torch.exp(log_var) + self.epsilon
        mu_sq = mu.pow(2)

        kl_loss = -0.5 * torch.mean(1 + log_var - mu_sq - var)
        scaled_kl_loss = self.lambda_kl * kl_loss

        # 4. Combine losses
        total_loss = mse_loss + angle_loss + scaled_kl_loss

        return total_loss
