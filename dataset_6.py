import torch
from torchvision.transforms import Compose, Grayscale, Normalize, ToTensor
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from losses import angles_table

device = torch.device("cuda")

class CrystalDataset(Dataset):
    def __init__(self, image_dir, std_dir, angles_table):
        self.image_dir = image_dir
        self.std_dir = std_dir
        self.angles_table = angles_table

        self.simulation_dirs = [d for d in os.listdir(image_dir)
                              if os.path.isdir(os.path.join(image_dir, d))]

        self.transform = Compose([
            Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(mean=[0.5], std=[0.5])
        ])

        self.all_files = []
        for sim_dir in self.simulation_dirs:
            sim_path = os.path.join(self.image_dir, sim_dir)
            png_files = [f for f in os.listdir(sim_path) if f.endswith('.png')]
            self.all_files.extend([(sim_dir, f) for f in sorted(png_files)])

        self.std_data = {}
        for sim_dir in self.simulation_dirs:
            std_path = os.path.join(self.std_dir, sim_dir, f"{sim_dir}.npy")
            if not os.path.exists(std_path):
                raise FileNotFoundError(f"Missing standard file: {std_path}")
            self.std_data[sim_dir] = torch.tensor(
                np.load(std_path),
                dtype=torch.float32
            )

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        sim_dir, filename = self.all_files[idx]
        
        image_path = os.path.join(self.image_dir, sim_dir, filename)
        image = self.transform(Image.open(image_path))

        std_intersections = self.std_data[sim_dir]

        # Angle table idx
        try:
            angle_idx = int(filename.split("_")[-1].split(".")[0])
            azimuth, elevation = self.angles_table[angle_idx]
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid angle index in {filename}: {e}")

        return (image,
                std_intersections.cpu(),
                torch.tensor(azimuth, dtype=torch.float32),
                torch.tensor(elevation, dtype=torch.float32))
