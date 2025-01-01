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


class HyperEnv(nn.Module):
    def __init__(self, target_shapes: dict):
        super(HyperEnv, self).__init__()
        self.target_shapes = target_shapes
        self.hidden_layer_bias = None
        self.hidden_layer_weight = None
        self.hidden_size = 32
        self.fc1_bias_gen = nn.Linear(self.hidden_size, self.target_shapes["fc1.bias"])
        self.fc1_weight_gen = nn.Linear(self.hidden_size, self.target_shapes["fc1.weight"])
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.hidden_layer_weight is None or self.hidden_layer_bias is None:
            self.hidden_layer_bias = nn.Linear(x.shape[1], self.hidden_size, device=x.device)
            self.hidden_layer_weight = nn.Linear(x.shape[1], self.hidden_size, device=x.device)
        x_bias = self.relu(self.hidden_layer_bias(x))
        x_weight = self.relu(self.hidden_layer_weight(x))
        bias = self.fc1_bias_gen(x_bias)
        weight = self.fc1_weight_gen(x_weight)
        return {'fc1.bias': bias, 'fc1.weight': weight}


class GammaRadHyperEnv(nn.Module):
    def __init__(self, shapes: dict):
        super(GammaRadHyperEnv, self).__init__()
        self.shapes = shapes
        self.gamma_rad_condition = GammaRadiationCondition(condition_dim=shapes["fc1.inp_dim"])
        target_shapes = {"fc1.bias": shapes["fc1.out_dim"], "fc1.weight": shapes["fc1.inp_dim"]*shapes["fc1.out_dim"]}
        self.environment_hypernet = HyperEnv(target_shapes=target_shapes)

    def forward(self, gamma: torch.Tensor, rad: torch.Tensor, env: torch.Tensor):
        x = self.gamma_rad_condition((gamma, rad))
        weights = self.environment_hypernet(env)
        x = torch.bmm(x.unsqueeze(1), weights['fc1.weight'].view(-1, self.shapes["fc1.inp_dim"], self. shapes["fc1.out_dim"])).squeeze(1) + weights['fc1.bias']
        return x


if __name__ == "__main__":
    model = GammaRadiationCondition()
    gamma = torch.randn(12, 502)
    radiation = torch.randn(12, 12, 46, 46)
    output = model((gamma, radiation))
    print(output.shape)
    env = torch.randn(12, 32)
    gammarad_condition_dim = 512
    antenna_dim = 40
    target_shapes = {"fc1.inp_dim": gammarad_condition_dim, "fc1.out_dim": antenna_dim}
    grhe = GammaRadHyperEnv(shapes=target_shapes)
    num_params_grhe = sum(p.numel() for p in grhe.parameters())
    print(f"Number of parameters in GammaRadHyperEnv: {num_params_grhe}")
    output_grhe = grhe((gamma, radiation), env)
