import torch
from torch import nn
import torch.nn.functional as F
import abc
# import AntennaDesign.pytorch_msssim as pytorch_msssim
import torchvision.models as models
from torch.autograd import Variable


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x, latent_vec=False):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class ResNet_VAE(AbstractAutoEncoder):
    def __init__(
            self, recon_loss_type=None, fc_hidden1=512,
            drop_p=0.3, CNN_embed_dim=512):
        super(ResNet_VAE, self).__init__()
        self.recon_loss_type = recon_loss_type
        self.fc_hidden1, self.CNN_embed_dim = fc_hidden1, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (5, 5), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (3, 3), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.resnet[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        # Latent vectors mu and sigma
        self.fc2_mu = nn.Linear(self.fc_hidden1, self.CNN_embed_dim)  # output = CNN embedding latent variables
        self.fc2_logvar = nn.Linear(self.fc_hidden1, self.CNN_embed_dim)  # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden1)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden1)
        self.fc5 = nn.Linear(self.fc_hidden1, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans9 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(512, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans10 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans11 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(128, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans12 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(64, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(1, momentum=0.01),  # y = (y1, y2, y3) \in [0 ,1]^3
        )

    def encode(self, x):
        self.shape = tuple(x.shape[2:])
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv
        # FC layers
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        mu, logvar = self.fc2_mu(x), self.fc2_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_bn4(self.fc4(z))
        x = self.relu(x)
        x = self.fc_bn5(self.fc5(x))
        x = self.relu(x).view(-1, 1024, 1, 1)

        x = self.convTrans9(x)
        x = self.convTrans10(x)
        x = self.convTrans11(x)
        x = self.convTrans12(x)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = 4 * torch.sigmoid(x) - 1
        x = F.interpolate(x, size=self.shape, mode='bilinear')
        return x

    def forward(self, x, latent_vec=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        if latent_vec:
            return x_reconst, mu, logvar, z
        else:
            return x_reconst, mu, logvar



class ResNet_CVAE(AbstractAutoEncoder):
    def __init__(
            self, recon_loss_type, fc_hidden1=1024,
            fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(ResNet_VAE, self).__init__()
        self.recon_loss_type = recon_loss_type
        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)  # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)  # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1,
                      padding=self.pd4),
            nn.BatchNorm2d(256, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.out_mu = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.out_logvar = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        # (256, 8, 8) -> (3, 256, 256)
        self.decoder = nn.Sequential(
            #  (256, 8, 8) -> (256, 16, 16)
            UpsamplingLayer(256, 256, activation="ReLU"),
            # -> (128, 32, 32)
            UpsamplingLayer(256, 128, activation="ReLU"),
            # -> (64, 64, 64)
            UpsamplingLayer(128, 64, activation="ReLU", type="upsample"),
            # -> (32, 128, 128)
            UpsamplingLayer(64, 32, activation="ReLU", bn=False),
            # -> (3, 256, 256)
            UpsamplingLayer(32, 3, activation="none", bn=False, type="upsample"),
            # nn.Tanh()
            nn.Hardtanh(-1.0, 1.0),
        )

    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = self.conv1(x)
        mu = self.out_mu(x)
        logvar = self.out_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, latent_vec=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z)
        if latent_vec:
            return x_reconst, mu, logvar, z
        else:
            return x_reconst, mu, logvar

    def loss_function(self, x, x_hat, mu, logvar):
        recon_loss = calc_reconstruction_loss(x, x_hat, self.recon_loss_type)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp())
        return kl_loss + recon_loss


class UpsamplingLayer(nn.Module):
    def __init__(self, in_channel, out_channel, activation="none", bn=True, type="transpose"):
        super(UpsamplingLayer, self).__init__()
        self.bn = nn.BatchNorm2d(out_channel) if bn else None
        if activation == "ReLU":
            self.activaton = nn.ReLU(True)
        elif activation == "none":
            self.activaton = None
        else:
            assert ()
        if type == "transpose":
            self.upsampler = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0),
            )
        elif type == "upsample":
            self.upsampler = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            )
        else:
            assert ()

    def forward(self, x):
        x = self.upsampler(x)
        if self.activaton:
            x = self.activaton(x)
        if self.bn:
            x = self.bn(x)
        return x