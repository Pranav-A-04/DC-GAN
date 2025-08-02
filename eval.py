import torch as th
import torch.nn as nn
import torch.functional as F
import numpy as np
from numpy import cov, iscomplexobj, trace
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import os
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from scipy.linalg import sqrtm
from models.generator import Generator
from models.discriminator import Discriminator

from torchmetrics.image import FrechetInceptionDistance

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)


latent_dim = config["latent_dim"]



def eval(generator, dataloader, device):
    generator.eval()
    
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    with th.no_grad():
        for real_imgs, _ in tqdm(dataloader, desc="Calculating FID"):
            real_imgs = real_imgs.to(device)
            noise = th.randn(real_imgs.size(0), latent_dim, device=device)
            fake_imgs = generator(noise)

            # Update FID metric
            fid_metric.update(real_imgs, real=True)
            fid_metric.update(fake_imgs, real=False)

    return fid_metric.compute().item()

def main():

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Load the generator and discriminator
    generator = Generator(
        latent_dim=latent_dim,
        im_size=32,
        im_channels=3,
        conv_channels=[1024, 512, 256, 128],
        kernels=[4, 4, 4, 4, 3],
        strides=[1, 2, 2, 2, 1],
        paddings=[0, 1, 1, 1, 1],
        output_paddings=[0, 0, 0, 0, 0]
    ).to(device)
    
    # Load the model ckpt
    ckpt_path = "checkpoints/generator_epoch_91.pth"  # <-- change this path if needed
    if os.path.exists(ckpt_path):
        generator.load_state_dict(th.load(ckpt_path, map_location=device))
        print(f"Loaded generator checkpoint from {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (2 * x) - 1)  # scale to [-1, 1]
    ])

    dataset = CIFAR10(root='data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Evaluate FID
    fid_score = eval(generator, dataloader, device)
    print(f"FID Score: {fid_score}")
    
if __name__ == "__main__":
    main()
        
        
            
            