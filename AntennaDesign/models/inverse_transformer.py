import torch
import torch.nn as nn
import math
from models import resnet
class inverse_transformer(nn.Module):
    def __init__(self, p_drop=0.25):
        super(inverse_transformer, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pe = self.positionalencoding2d(d_model=128, height=8, width=2).to(device)
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4, 16), resnet.ResNetBasicBlock(16, 16),
                                                resnet.ResNetBasicBlock(16, 16), resnet.ResNetBasicBlock(16, 16),
                                                nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(64, 128, kernel_size=2), nn.BatchNorm2d(128), self.relu)
        self.fc1 = nn.Linear(2550, 1024)
        transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 12)

    def forward(self, input):
        gamma, radiation = input
        if radiation.ndim == 3:
            radiation = radiation.unsqueeze(0)
        radiation_features = self.radiation_backbone(radiation)
        radiation_features = radiation_features + self.pe
        radiation_features = radiation_features.permute(0, 2, 3, 1)
        radiation_features = radiation_features.contiguous().view(radiation_features.shape[0], -1, radiation_features.shape[3])
        transformer_features = self.transformer_encoder(radiation_features)
        transformer_features = transformer_features.view(transformer_features.shape[0], -1)
        x = torch.cat((gamma, transformer_features), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x
    def positionalencoding2d(self,d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe
