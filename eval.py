import numpy as np
import torch as th
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import make_grid
import os
import yaml
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import csv
from torchmetrics.image.fid import FrechetInceptionDistance
import yaml

from models.generator import Generator
from models.discriminator import Discriminator

with open("DC-GAN/config.yml", "r") as f:
    config = yaml.safe_load(f)


latent_dim = config["latent_dim"]

training_config = config['train_params']

batch_size = training_config['batch_size']
num_grid_rows = training_config['num_grid_rows']
learning_rate = training_config['lr']
num_samples = training_config['num_samples']
num_epochs = training_config['num_epochs']

device = th.device("cuda" if th.cuda.is_available() else "cpu")

def sample(epoch, generator, device):
    with th.no_grad():
        fake_im_noise = th.randn((num_samples, latent_dim), device=device)
        fake_ims = generator(fake_im_noise)
        fake_ims = (fake_ims + 1) / 2
        grid = make_grid(fake_ims, nrow=num_grid_rows)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists('samples'):
            os.mkdir('samples')
        img.save(os.path.join('samples', '{}.png'.format(epoch)))

def train(generator, discriminator, loss_fn, optimizer_disc, optimizer_gen, dataloader, device, epochs=100):
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    generator.train()
    discriminator.train()
    # Prepare CSV file for logging losses
    csv_file = 'loss_log.csv'
    write_header = not os.path.exists(csv_file)
    fid_header = not os.path.exists("fid_scores_training.csv")
    with open(csv_file, mode='a', newline='') as loss_f, \
        open('fid_scores_training.csv', mode='a', newline='') as fid_f:
        loss_writer = csv.writer(loss_f)
        fid_writer = csv.writer(fid_f)
        if write_header:
            loss_writer.writerow(['epoch', 'generator_loss', 'discriminator_loss'])
        
        if fid_header:
            fid_writer.writerow(["epoch", "FID"])
            
        for epoch in range(epochs):
            generator_losses=[]
            discriminator_losses = []
            mean_real_dis_preds = []
            mean_fake_dis_preds = []
            fid_score = 99999  # Initialize FID score
            for im, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                real_imgs = im.float().to(device)
                batch_size = real_imgs.shape[0]
                
                #optimize discriminator
                optimizer_disc.zero_grad()
                z = th.randn(batch_size, latent_dim).to(device)
                fake_imgs = generator(z)
                
                # label smoothing
                real_labels = th.full((batch_size,), 0.9, device=device)
                fake_labels = th.full((batch_size,), 0.1, device=device)
            
                real_preds = discriminator(real_imgs)
                fake_preds = discriminator(fake_imgs)
                
                disc_real_loss = loss_fn(real_preds.reshape(-1), real_labels.reshape(-1))
                disc_fake_loss = loss_fn(fake_preds.reshape(-1), fake_labels.reshape(-1))
                mean_real_dis_preds.append(th.nn.Sigmoid()(real_preds).mean().item())
                mean_fake_dis_preds.append(th.nn.Sigmoid()(fake_preds).mean().item())
                disc_loss = (disc_real_loss + disc_fake_loss) / 2
                
                disc_loss.backward()
                optimizer_disc.step()
                
                #optimize generator (2 times per iteration)
                for _ in range(2):
                    optimizer_gen.zero_grad()
                    z = th.randn(batch_size, latent_dim, device=device)
                    fake_imgs = generator(z)
                    fake_preds = discriminator(fake_imgs)
                    gen_loss = loss_fn(fake_preds.reshape(-1), real_labels.reshape(-1))
                    gen_loss.backward()
                    optimizer_gen.step()
                    generator_losses.append(gen_loss.item())  # Log each gen update
                
                # store for FID computation
                if (epoch + 1) % 5 == 0:
                    fake_imgs_fid = ((fake_imgs + 1) / 2).detach()
                    real_imgs_fid = ((real_imgs + 1) / 2).detach()
                    
                    # Compute FID score
                    generator.eval()
                    with th.no_grad():
                        fid_metric.update(real_imgs_fid, real=True)
                        fid_metric.update(fake_imgs_fid, real=False)

                    generator.train()
                
                discriminator_losses.append(disc_loss.item())
                #generator_losses.append(gen_loss.item())            
            
            # Save samples
            if (epoch+1) % 5 == 0:
                generator.eval()
                sample(epoch, generator, device)
                generator.train()
                
                # Compute FID score
                fid_score = fid_metric.compute().item()
                fid_metric.reset()
            
            # Log losses to CSV
            loss_writer.writerow([
                epoch + 1,
                np.mean(generator_losses),
                np.mean(discriminator_losses)
            ])
            
            # Print epoch summary
            print('Finished epoch:{} | Generator Loss : {:.4f} | Discriminator Loss : {:.4f}| '
                  'Discriminator real pred : {:.4f} | Discriminator fake pred : {:.4f} | FID Score: {:.4f}'.format(
                    epoch + 1,
                    np.mean(generator_losses),
                    np.mean(discriminator_losses),
                    np.mean(mean_real_dis_preds),
                    np.mean(mean_fake_dis_preds),
                    fid_score if fid_score != 99999 else 0.0  # FID score might not be computed for this epoch
                    )
                )
            
            
            
            # Save model checkpoints and write FID scores to csv
            if (epoch+1) % 5 == 0:
                fid_writer.writerow([epoch+1, fid_score])
                if not os.path.exists('checkpoints'):
                    os.mkdir('checkpoints')
                th.save(generator.state_dict(), os.path.join('checkpoints', f'generator_epoch_{epoch+1}.pth'))
                th.save(discriminator.state_dict(), os.path.join('checkpoints', f'discriminator_epoch_{epoch+1}.pth'))

def main():
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
    
    discriminator = Discriminator(
        im_size=32,
        im_channels=3,  
        conv_channels=[128, 256, 512, 1024],
        kernels=[3, 4, 4, 4, 4],
        strides=[1, 2, 2, 2, 2],
        paddings=[1, 1, 1, 1, 0]
    ).to(device)
    
    # Use DataParallel if multiple GPUs are available
    # if th.cuda.device_count() > 1:
    #     print(f"Using {th.cuda.device_count()} GPUs for training.")
        
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: (2 * x) - 1)  # scale to [-1, 1]
                ])
    dataset = CIFAR10(root='data', train=True, download=True, transform=transform)
    print("downloaded CIFAR10 dataset")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    loss_fn = th.nn.BCEWithLogitsLoss()
    optimizer_disc = th.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_gen = th.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    
    train(generator, discriminator, loss_fn, optimizer_disc, optimizer_gen, dataloader, device, epochs=num_epochs)

if __name__ == "__main__":
    main()
            
            