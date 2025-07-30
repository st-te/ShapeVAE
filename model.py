import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda")  # Default to CUDA

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.flat_size = 256 * 8 * 8  # For 120x120 input

        # separate paths for orientation and shape features
        self.orientation_fc = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 32),
            nn.LeakyReLU()
        )

        self.shape_fc = nn.Sequential(
            nn.Linear(self.flat_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU()
        )

        # final latent projections
        self.fc_mu_orientation = nn.Linear(32, 2)  # first 2 dims for azimuth/elevation
        self.fc_logvar_orientation = nn.Linear(32, 2)
        self.fc_mu_shape = nn.Linear(256, latent_dim - 2)  # remaining dims for shape
        self.fc_logvar_shape = nn.Linear(256, latent_dim - 2)

    def forward(self, x):
        batch_size = x.size(0)
        if x.size() != (batch_size, 1, 120, 120):
            raise ValueError(f"Expected input size (N, 1, 120, 120), got {x.size()}")
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)

        # process orientation features
        orientation_features = self.orientation_fc(x)
        mu_orientation = self.fc_mu_orientation(orientation_features)
        logvar_orientation = self.fc_logvar_orientation(orientation_features)

        # process shape features
        shape_features = self.shape_fc(x)
        mu_shape = self.fc_mu_shape(shape_features)
        logvar_shape = self.fc_logvar_shape(shape_features)

        # concatenate orientation and shape parameters
        mu = torch.cat([mu_orientation, mu_shape], dim=1)
        logvar = torch.cat([logvar_orientation, logvar_shape], dim=1)

        return mu, logvar


class CNNDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(CNNDecoder, self).__init__()

        self.orientation_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16)
        )

        # prediction heads for angles
        self.azimuth_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )    # prediction for [0, 360]
        self.elevation_head = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )   # prediction for [0, 180]

        # shape processing
        self.shape_fc = nn.Sequential(
            nn.Linear(latent_dim-2, 1024),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3)
        )

        # 3D reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(2048, 706),  # output size for 3D representation
            nn.Sigmoid(),
            nn.Hardtanh(min_val=0.0, max_val=2.0)  # ensure [0, 2] range
        )

    def forward(self, z):
        """
        Forward pass that handles both orientation prediction and 3D reconstruction

        Args:
            z (tensor): Latent vector where:
                - z[:, :2] contains orientation information
                - z[:, 2:] contains shape information

        Returns:
            tuple: (reconstruction, azimuth, elevation) where:
                - reconstruction is the 3D structure (706-dimensional)
                - azimuth is the predicted viewing angle [0, 360]
                - elevation is the predicted viewing angle [0, 180]
        """
        # split latent vector
        z_orientation = z[:, :2]    # orientation components
        z_shape = z[:, 2:]         # shape components

        orientation_features = self.orientation_net(z_orientation)
        azimuth = self.azimuth_head(orientation_features) * 360.0
        elevation = self.elevation_head(orientation_features) * 180.0

        shape_features = self.shape_fc(z_shape)
        reconstruction = self.reconstruction_head(shape_features)

        return reconstruction, azimuth.squeeze(1), elevation.squeeze(1)


class ShapeVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(ShapeVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = CNNDecoder(latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std, dtype=torch.float32, device=std.device)
        return mu + eps * std

    def forward(self, x):
        x = x.to(next(self.parameters()).device).float()
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed, azimuth, elevation = self.decoder(z)
        return x_reconstructed, azimuth, elevation, mu, log_var
