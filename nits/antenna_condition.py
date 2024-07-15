import torch
import torch.nn as nn
from nits import resnet


class gamma_radiation_condition(nn.Module):
    def __init__(self, p_drop=0.25, condition_dim=12):
        super(gamma_radiation_condition, self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.radiation_backbone_input_layer = None
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(16, 16),
                                                resnet.ResNetBasicBlock(16, 16), resnet.ResNetBasicBlock(16, 16),
                                                nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(256, 286, kernel_size=3), nn.BatchNorm2d(286), self.relu)
        self.fc1 = nn.Linear(2504, 1024)
        self.fc2 = nn.Linear(1024, condition_dim)

    def forward(self, input):
        gamma, radiation = input
        if radiation.ndim == 3:
            radiation = radiation.unsqueeze(0)
        if self.radiation_backbone_input_layer is None:
            self.radiation_backbone_input_layer = resnet.ResNetBasicBlock(radiation.shape[1], 16)
        radiation_inp = self.radiation_backbone_input_layer(radiation)
        radiation_features = self.radiation_backbone(radiation_inp)
        radiation_features = radiation_features.view(radiation_features.shape[0], -1)
        x = torch.cat((gamma, radiation_features), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
