import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.vae import VAE


def plot_samples(samples, samples_per_row=8):
    n_rows = len(samples) // samples_per_row
    n_tot = int(n_rows*samples_per_row)
    samples = samples[:n_tot]
    fig = plt.figure(figsize=(2*samples_per_row, 2*n_rows))
    for i, out in enumerate(samples):
        a = fig.add_subplot(n_rows, samples_per_row, i+1)
        plt.imshow(out, cmap='binary')
        a.axis("off")

    plt.show()

def get_n_params(model):
    return np.sum([np.prod(param.shape) for param in model.parameters()])

if __name__ == "__main__":

    input_shape = (1, 28, 28)
    latent_dim = 64

    architecture = 'linear'

    # load model
    model = VAE(input_shape, latent_dim, architecture=architecture)
    loaded_state_dict = torch.load(f"saved_models/vae_{architecture}.pt", map_location=torch.device('cpu'))
    model.load_state_dict(loaded_state_dict)
    model.eval()

    print(f"Total number of parameters for {architecture} architecture: {get_n_params(model):d}")

    x_hat = model.sample(n=16).detach().numpy().reshape(-1, 28, 28)

    plot_samples(x_hat, samples_per_row=8)



