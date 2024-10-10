import torch
import torch.nn as nn
from models import baseline_regressor, forward_radiation


class forward_GammaRad(nn.Module):
    def __init__(self, radiation_channels, gamma_model=None, radiation_model=None, rad_range=None):
        super(forward_GammaRad, self).__init__()
        if rad_range is None:
            rad_range = [-20, 5]
        if gamma_model is None:
            self.gamma_net = baseline_regressor.small_deeper_baseline_forward_model()
        else:
            self.gamma_net = gamma_model

        if radiation_model is None:
            self.radiation_net = forward_radiation.Radiation_Generator(radiation_channels, rad_range)
        else:
            self.radiation_net = radiation_model

    def forward(self, x):
        gamma_pred = self.gamma_net(x)
        radiation_pred = self.radiation_net(x)
        return gamma_pred, radiation_pred

    def load_and_freeze_forward(self, weights_path):
        path_rad, path_gamma = weights_path
        if path_gamma is not None:
            self.gamma_net.load_state_dict(torch.load(path_gamma))
        if path_rad is not None:
            self.radiation_net.load_state_dict(torch.load(path_rad))
        for param in self.parameters():
            param.requires_grad = False
