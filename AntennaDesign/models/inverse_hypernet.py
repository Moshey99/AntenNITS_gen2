import torch
import torch.nn as nn
from models import resnet, baseline_regressor


class inverse_radiation_no_hyper(nn.Module):
    def __init__(self, p_drop=0.25):
        super(inverse_radiation_no_hyper, self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4, 16), resnet.ResNetBasicBlock(16, 16),
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
        self.fc1 = nn.Linear(4004, 1024)
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        # self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 12)


class small_inverse_radiation_no_hyper(nn.Module):
    def __init__(self, p_drop=0.25):
        super(small_inverse_radiation_no_hyper, self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4, 16), resnet.ResNetBasicBlock(16, 16),
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
        # self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        # self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 128)
        # self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 12)

    def forward(self, input):
        gamma, radiation = input
        if radiation.ndim == 3:
            radiation = radiation.unsqueeze(0)
        radiation_features = self.radiation_backbone(radiation)
        radiation_features = radiation_features.view(radiation_features.shape[0], -1)
        x = torch.cat((gamma, radiation_features), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class inverse_radiation_hyper(nn.Module):
    def __init__(self, p_drop=0.25):
        super(inverse_radiation_hyper, self).__init__()
        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=p_drop)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.hyper_fc_shapes = [(64, 32), (32, 32), (32, 12)]
        self.hyper_alternative = nn.ModuleList([nn.Linear(64, 32), nn.Linear(32, 32), nn.Linear(32, 12)])
        self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4, 16), resnet.ResNetBasicBlock(16, 16),
                                                resnet.ResNetBasicBlock(16, 16), resnet.ResNetBasicBlock(16, 16),
                                                nn.Conv2d(16, 32, kernel_size=3), nn.BatchNorm2d(32), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), self.relu,
                                                self.maxpool,
                                                nn.Conv2d(256, 512, kernel_size=3), nn.BatchNorm2d(512))
        total_parameters = 0
        for i, fc_shape in enumerate(self.hyper_fc_shapes):
            in_size, out_size = fc_shape
            cnt = in_size * out_size + 2 * out_size
            total_parameters += cnt
        self.radiation_hyper_fc = nn.Linear(3584, total_parameters)
        nn.init.zeros_(self.radiation_hyper_fc.bias)
        self.init_weights_hyper(self.radiation_hyper_fc.weight, dj=32, var_e=12.38)
        self.fcs = nn.Sequential(nn.Linear(502, 256), self.relu, nn.Linear(256, 128), self.relu, nn.Linear(128, 64))

    def init_weights_hyper(self, weight, dj, var_e):
        with torch.no_grad():
            dk = weight.shape[1]  # fan_in of last fc hypernetwork layer
            std_weight = torch.sqrt(torch.tensor(1 / (dk * dj * var_e)))
            return weight.normal_(0, std_weight)

    def forward(self, input):
        gamma, radiation = input
        radiation_features = self.radiation_backbone(radiation)
        radiation_features = radiation_features.view(-1)
        radiation_features = self.radiation_hyper_fc(radiation_features)  # activation needed?
        x = self.relu(self.fcs(gamma))
        passed = 0
        for i, fc_shape in enumerate(self.hyper_fc_shapes):
            in_size, out_size = fc_shape
            current_amount = in_size * out_size + out_size * 2
            current_parameters = radiation_features[passed:passed + current_amount]
            passed += current_amount
            weights = current_parameters[:in_size * out_size].view(out_size, in_size)
            scale = current_parameters[in_size * out_size:in_size * out_size + out_size]
            bias = current_parameters[in_size * out_size + out_size:in_size * out_size + out_size * 2]
            if i == len(self.hyper_fc_shapes) - 1:
                x = self.hyper_alternative[i](x)
            else:
                x = self.hyper_alternative[i](x)
                x = self.relu(x)

        return x

    # class inverse_radiation_hyper(nn.Module):
    #     def __init__(self, p_drop=0.25):
    #         super(inverse_radiation_hyper,self).__init__()
    #         self.relu = nn.ELU()
    #         self.dropout = nn.Dropout(p=p_drop)
    #         self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
    #         self.hyper_fc_shapes = [(64,32),(32,32),(32,12)]
    #         self.hyper_biases = [nn.Parameter(torch.zeros(1,shape[1])) for shape in self.hyper_fc_shapes]
    #         self.hyper_weights = [nn.Parameter(torch.zeros(shape)) for shape in self.hyper_fc_shapes]
    #         for bias in self.hyper_biases:
    #             nn.init.kaiming_uniform_(bias)
    #         for weight in self.hyper_weights:
    #             nn.init.kaiming_uniform_(weight)
    #         self.radiation_backbone = nn.Sequential(resnet.ResNetBasicBlock(4,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),resnet.ResNetBasicBlock(16,16),
    #                                                 nn.Conv2d(16,32,kernel_size=3),nn.BatchNorm2d(32),self.relu,self.maxpool,
    #                                                 nn.Conv2d(32,64,kernel_size=3),nn.BatchNorm2d(64),self.relu,self.maxpool,
    #                                                 nn.Conv2d(64,128,kernel_size=3),nn.BatchNorm2d(128),self.relu,self.maxpool,
    #                                                 nn.Conv2d(128,256,kernel_size=3),nn.BatchNorm2d(256),self.relu,self.maxpool,
    #                                                 nn.Conv2d(256,512,kernel_size=3),nn.BatchNorm2d(512))
    #         total_parameters = 0
    #         for i,fc_shape in enumerate(self.hyper_fc_shapes):
    #             in_size,out_size = fc_shape
    #             cnt = in_size*out_size
    #             total_parameters += cnt
    #         self.radiation_hyper_fc = nn.Linear(3584,total_parameters)
    #         self.init_weights_hyper(self.radiation_hyper_fc.weight,dj=32,var_e=12.38)
    #         self.init_bias_hyper(self.radiation_hyper_fc.bias,var_e=12.38)
    #         self.fcs = nn.Sequential(nn.Linear(502,256),self.relu,nn.Linear(256,128),self.relu,nn.Linear(128,64))
    #
    #     def init_weights_hyper(self,weight,dj,var_e):
    #         with torch.no_grad():
    #             dk = weight.shape[1] # fan_in of last fc hypernetwork layer
    #             std_weight = torch.sqrt(torch.tensor(1/(dk*dj*var_e)))
    #             return weight.normal_(0,std_weight)
    #
    #
    #
    #     def forward(self,input):
    #         gamma, radiation = input
    #         radiation_features = self.radiation_backbone(radiation)
    #         radiation_features = radiation_features.view(-1)
    #         radiation_features = self.radiation_hyper_fc(radiation_features) # activation needed?
    #         x = self.relu(self.fcs(gamma))
    #         for i,fc_shape in enumerate(self.hyper_fc_shapes):
    #             in_size,out_size = fc_shape
    #             weights = self.hyper_weights[i] #radiation_features[:in_size*out_size].view(out_size,in_size)
    #             bias = self.hyper_biases[i]
    #
    #             if i == len(self.hyper_fc_shapes)-1:
    #                 x = self.scaled_linear(weights,bias,input=x)
    #             else:
    #                 x = self.scaled_linear(weights,bias,input=x)
    #                 x = self.relu(x)
    #                 x = nn.LayerNorm(x.shape[1:])(x) # to make sure var(x) = 1
    #
    #         return x

    def scaled_linear(self, weights, scale, bias, input):
        input = input.matmul(weights.t()) * scale + bias
        return input


class inverse_forward_concat(nn.Module):
    def __init__(self, inv_module=None, forw_module=None, forward_weights_path_rad=None,
                 forward_weights_path_gamma=None):
        super(inverse_forward_concat, self).__init__()
        if inv_module is None:
            self.inverse_module = small_inverse_radiation_no_hyper()
        else:
            self.inverse_module = inv_module
        if forw_module is None:
            self.forward_module = baseline_regressor.small_deeper_baseline_forward_model()
        else:
            self.forward_module = forw_module
        if forward_weights_path_rad is not None or forward_weights_path_gamma is not None:
            self.load_and_freeze_forward((forward_weights_path_rad, forward_weights_path_gamma))

    def load_and_freeze_forward(self, weights_path):
        path_rad, path_gamma = weights_path
        if path_gamma is not None:
            self.forward_module.gamma_net.load_state_dict(torch.load(path_gamma))
        if path_rad is not None:
            self.forward_module.radiation_net.load_state_dict(torch.load(path_rad))
        for param in self.forward_module.parameters():
            param.requires_grad = False

    def forward(self, input):
        geometry = self.inverse_module(input)
        gamma, radiation = self.forward_module(geometry)
        return gamma, radiation, geometry


if __name__ == "__main__":
    encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=208)
    src = torch.rand(10, 32, 512)
    out = encoder_layer(src)
    print(out.shape)
