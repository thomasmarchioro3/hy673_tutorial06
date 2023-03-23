import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import os
from tqdm import tqdm

from utils.vae import VAE

def loss_fn(x, x_hat, logvar):
    # Reconstruction loss (could use also MSE, but BCE is good for inputs bounded in [0, 1])
    bce = F.binary_cross_entropy(x_hat, x, reduction='mean')

    # KLD between Gaussian rvs
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

if __name__ == "__main__":

    # define hyper parameters
    batch_size = 128
    epochs = 1  # 20
    lr = 1e-3

    architecture = 'linear'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    torch.manual_seed(42)

    input_shape = (1, 28, 28)
    latent_dim = 64

    # load model

    model = VAE(input_shape, latent_dim, architecture=architecture)
    model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load the MNIST dataset
    train_dataset = MNIST(root='./data', train=True, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    losses = []
    best_loss = 100    

    if not os.path.isdir('saved_models'):
        os.makedirs('saved_models')

    for epoch in range(epochs):

        train_loss = 0
        for x, _ in tqdm(train_loader):
            
            optimizer.zero_grad()

            x = x.to(device)
            x_hat, mu, logvar, z = model(x)

            loss = loss_fn(x, x_hat, logvar)
            train_loss += loss.item()
            loss.backward()


            optimizer.step()
 
        train_loss /= len(train_loader)
        print(f"Average loss per batch at epoch {epoch+1:03d}: {train_loss:.4f}")
        losses.append(train_loss)
        
        # save model if it improved
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), f'saved_models/vae_{architecture}.pt')

            


