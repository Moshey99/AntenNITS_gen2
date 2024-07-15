import torch
import torch.nn as nn
import numpy as np
class Gamma_Generator(nn.Module):
    def __init__(self):
        super(Gamma_Generator,self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.input_layer = nn.Sequential(
            nn.Linear(12, 64),
            nn.ELU()
        )
        self.layers = nn.Sequential(nn.ConvTranspose1d(1,2,kernel_size=4,stride=2,padding=1),
                                    nn.ELU(),
                                    nn.ConvTranspose1d(2,2,kernel_size=3,stride=2,padding=1),
                                    nn.ELU(),
                                    nn.Conv1d(2,2,kernel_size=5,stride=1),
                                    nn.ELU(),
                                    # nn.ConvTranspose1d(1,1,kernel_size=4,stride=2,padding=1),
                                    # nn.ELU(),
                                    # nn.Conv1d(1,1,kernel_size=3,stride=1),
                                    )


    def forward(self, x):
        x = self.input_layer(x)
        x = x.view(x.size(0), 1,-1)
        x = self.layers(x)
        x[:,0,:] = self.sigmoid(x[:,0,:])
        x = torch.cat((x[:,0,:],x[:,1,:]),dim=1)
        return x

if __name__ == '__main__':
    generator = Gamma_Generator()
    input_features = torch.randn(20, 12)
    output_image = generator(input_features)
    print(output_image.shape)