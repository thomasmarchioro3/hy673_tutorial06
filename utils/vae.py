import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np



class VAE(nn.Module):

    def __init__(self, input_shape:tuple=(1, 28, 28), latent_dim:int=32, 
                 architecture:str='linear'):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.input_dim = np.prod(input_shape)
        self.latent_dim = latent_dim

        if architecture == 'linear':
            self.encoder = LinearEncoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
            self.decoder = LinearDecoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        elif architecture == 'conv':
            self.encoder = ConvEncoder(input_shape=self.input_shape, latent_dim=self.latent_dim)
            self.decoder = ConvDecoder(input_shape=self.input_shape, latent_dim=self.latent_dim)
        else:
            raise Exception(f'Error: No "{architecture}" architecture.')


    def forward(self, x):
        x = x.view(-1, self.input_dim)
        mu, logvar = self.encoder(x)
        z = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z).view(-1, *self.input_shape)

        return x_hat, mu, logvar, z
    
    def sample(self, n: int):

        z = torch.randn(n, self.latent_dim)
        return self.decoder(z).view(-1, *self.input_shape)
        
    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(.5*logvar)
        z = torch.randn_like(std)

        return std * z + mu
    

class LinearEncoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int=32):
        super(LinearEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * 2)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # no activation for the last layer

        # half of the latent values are used for the mean, half for the log-variance
        mu = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:] 
        return mu, logvar

class LinearDecoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int=32):
        super(LinearDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.rev_fc3 = nn.Linear(self.latent_dim, 256)
        self.rev_fc2 = nn.Linear(256, 512)
        self.rev_fc1 = nn.Linear(512, self.input_dim)

    def forward(self, z):
        x = torch.relu(self.rev_fc3(z))
        x = torch.relu(self.rev_fc2(x))
        x = torch.sigmoid(self.rev_fc1(x))
        return x
    

class ConvEncoder(nn.Module):

    def __init__(self, input_shape:tuple=(1, 28, 28), latent_dim:int=32):
        super(ConvEncoder, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        
        # determine the output shapes for conv layers
        self.outconv1_shape = conv2Doutshape(self.input_shape, 16, kernel_size=4, stride=2, padding=1)
        self.outconv2_shape = conv2Doutshape(self.outconv1_shape, 32, kernel_size=4, stride=2, padding=1)

        # linear layers
        self.outconv_dim = np.prod(self.outconv2_shape)
        self.fc = nn.Linear(self.outconv_dim, 2*self.latent_dim)


    def forward(self, x):

        # x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        x = x.view(-1, *self.input_shape)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # flatten
        x = x.view(-1, self.outconv_dim)
        x = self.fc(x)

        mu = x[:, :self.latent_dim]
        logvar = x[:, self.latent_dim:] 

        return mu, logvar
    

class ConvDecoder(nn.Module):

    def __init__(self, input_shape:tuple=(1, 28, 28), latent_dim:int=32):
        super(ConvDecoder, self).__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # determine the output shapes for conv layers
        self.outconv1_shape = conv2Doutshape(self.input_shape, 16, kernel_size=4, stride=2, padding=1)
        self.outconv2_shape = conv2Doutshape(self.outconv1_shape, 32, kernel_size=4, stride=2, padding=1)

        # define convtranspose layers
        self.convt2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.convt1 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        # linear layers
        self.outconv_dim = np.prod(self.outconv2_shape)
        self.rev_fc = nn.Linear(self.latent_dim, self.outconv_dim)

    def forward(self, z):

        x = torch.relu(self.rev_fc(z))  # 32*7*7
        x = x.view(-1, *self.outconv2_shape)  # (32, 7, 7)
        x = torch.relu(self.convt2(x))  # (16, 14, 14)
        x = torch.sigmoid(self.convt1(x))  # (1, 28, 28)
        return x


"""
Get the output_shape of a Conv2D layer given the input_shape and parameters.
"""
def conv2Doutshape(input_shape:tuple, filters, kernel_size, stride=1, padding=0, dilation=1):

    if isinstance(kernel_size, (float, int)):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, (float, int)):
        stride = (stride, stride)
    if isinstance(padding, (float, int)):
        padding = (padding, padding)
    if isinstance(dilation, (float, int)):
        dilation = (dilation, dilation)

    _, h_in, w_in = input_shape

    # formula from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    c_out = filters
    h_out = int( (h_in + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) -1)/stride[0] + 1)
    w_out = int( (w_in + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) -1)/stride[1] + 1)

    return c_out, h_out, w_out


if __name__ == "__main__":

    # here it is a good idea to put testing/debugging code

    input_shape = (1, 28, 28)
    latent_dim = 20


    model = VAE(
        input_shape=input_shape,
        latent_dim=latent_dim,
        architecture='conv'
    )


    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

    x, _ = next(iter(train_loader))
    x_hat, mu, logvar, z = model(x)

    exit(0)