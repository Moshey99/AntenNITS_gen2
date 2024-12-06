import torch
import torch.nn as nn
from nits import resnet


class GammaRadiationCondition(nn.Module):
    def __init__(self, p_drop=0.25, condition_dim=12):
        super(GammaRadiationCondition, self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.radiation_backbone_input_layer = None
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(16, 16), resnet.ResNetBasicBlock(16, 16),
                                                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(32), self.relu, self.maxpool,
                                                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(64), self.relu, self.maxpool,
                                                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(128), self.relu, self.maxpool,
                                                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(256), self.relu, self.maxpool,
                                                nn.Conv2d(256, 500, kernel_size=3, padding=1),
                                                nn.BatchNorm2d(500), self.relu)

        self.fc1 = nn.Linear(2502, 1024)
        self.fc2 = nn.Linear(1024, condition_dim)

    def forward(self, input):
        gamma, radiation = input
        if radiation.ndim == 3:
            radiation = radiation.unsqueeze(0)
        rad_sep = radiation.shape[1] // 2
        radiation = radiation[:, :rad_sep]  # only use magnitude of radiation for condition
        if self.radiation_backbone_input_layer is None:
            self.radiation_backbone_input_layer = resnet.ResNetBasicBlock(radiation.shape[1], 16).to(radiation.device)
        radiation_inp = self.radiation_backbone_input_layer(radiation)
        radiation_features = self.radiation_backbone(radiation_inp)
        radiation_features = radiation_features.view(radiation_features.shape[0], -1)
        x = torch.cat((gamma, radiation_features), dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class EnvironmentCondition(nn.Module):
    def __init__(self, output_dim=12):
        super(EnvironmentCondition, self).__init__()
        self.relu = nn.ELU()
        self.env_input_layer = None
        self.output_dim = output_dim
        self.output_layer = nn.Linear(self.output_dim, self.output_dim)

    def forward(self, env):
        if self.env_input_layer is None:
            self.env_input_layer = nn.Linear(env.shape[1], self.output_dim).to(env.device)
        x = self.relu(self.env_input_layer(env))
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    model = GammaRadiationCondition()
    gamma = torch.randn(1, 502)
    radiation = torch.randn(1, 12, 46, 46)
    output = model((gamma, radiation))
    print(output.shape)
