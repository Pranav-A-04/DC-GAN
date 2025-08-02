import torch as th
from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import os
import yaml
import argparse
import csv
import re

from models.generator import Generator

# Load config
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

            fid_metric.update(real_imgs, real=True)
            fid_metric.update(fake_imgs, real=False)

    return fid_metric.compute().item()


def extract_epoch(ckpt_path):
    #generator_epoch_91.pth
    match = re.search(r'epoch_(\d+)', ckpt_path)
    return int(match.group(1)) if match else -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to generator checkpoint')
    parser.add_argument('--csv', type=str, default='fid_scores.csv', help='CSV output file')
    args = parser.parse_args()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # Load generator
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

    # Load checkpoint
    if os.path.exists(args.ckpt):
        generator.load_state_dict(th.load(args.ckpt, map_location=device))
        print(f"Loaded generator checkpoint from {args.ckpt}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (2 * x) - 1)
    ])

    dataset = CIFAR10(root='data', train=False, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Evaluate
    fid_score = eval(generator, dataloader, device)
    epoch = extract_epoch(args.ckpt)
    print(f"Epoch {epoch} -> FID Score: {fid_score:.4f}")

    # Write to CSV
    file_exists = os.path.exists(args.csv)
    with open(args.csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["epoch", "FID"])
        writer.writerow([epoch, fid_score])


if __name__ == "__main__":
    main()
        
            
            