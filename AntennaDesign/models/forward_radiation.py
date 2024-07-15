import torch
import torch.nn as nn
import numpy as np

class Radiation_Generator(nn.Module):
    def __init__(self, radiation_channels, radiation_range=[-55, 5]):
        super(Radiation_Generator, self).__init__()
        self.activation = nn.ELU()
        self.length = radiation_range[1]-radiation_range[0]
        self.radiation_range = radiation_range
        self.radiation_channels = radiation_channels
        # Input layer for geometrical features
        self.input_layer = None
        self.sigmoid = nn.Sigmoid()
        # Transpose convolutional layers to upsample the input
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ELU(),
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=2),
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ELU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(4, 4, kernel_size=3),
            nn.ELU())
        self.output_layer = None

    def forward(self, x):
        if self.input_layer is None:
            self.input_layer = nn.Sequential(nn.Linear(x.shape[1], 64), nn.ELU())
        x = self.input_layer(x)
        x = x.view(x.size(0), 64, 1, 1)  # Reshape to match the convolutional layers
        x = self.layers(x)
        if self.output_layer is None:
            self.output_layer = nn.ConvTranspose2d(4, self.radiation_channels, kernel_size=3, stride=1, padding=1)
        x = self.output_layer(x)
        sep = x.shape[1]//2
        x[:,:sep,:,:] = self.sigmoid(x[:,:sep,:,:])*self.length + self.radiation_range[0] # Normalize to radiation_range
        x[:,sep:,:,:] = self.sigmoid(x[:,sep:,:,:])*2*np.pi-np.pi # Normalize to [-pi,pi]
        return x


if __name__ == '__main__':
    generator = Radiation_Generator(radiation_channels=2)
    input_features = torch.randn(1, 10)
    output_image = generator(input_features)
    print(output_image.shape)
