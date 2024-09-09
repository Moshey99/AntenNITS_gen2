import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from AntennaDesign.utils import AntennaDataSetsLoader
import argparse
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """

    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
        self.relu = nn.ReLU()
        self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in

    def forward(self, x):
        conv1 = self.BN(self.conv1(x))
        relu = self.relu(conv1)
        conv2 = self.BN(self.conv2(relu))
        if self.resize:
            x = self.BN(self.conv1(x))
        return self.relu(x + conv2)


class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(1, 8, 5, 1, 2)
        self.BN = nn.BatchNorm2d(8)
        self.rb1 = ResBlock(8, 16, 3, 1, 1, 'encode')
        self.rb2 = ResBlock(16, 32, 3, 1, 1, 'encode')
        self.rb3 = ResBlock(32, 64, 3, 1, 1, 'encode')
        self.rb4 = ResBlock(64, 128, 3, 1, 1, 'encode')
        self.rb5 = ResBlock(128, 256, 3, 1, 1, 'encode')
        self.rb6 = ResBlock(256, 333, 3, 1, 1, 'encode')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, inputs):
        init_conv = self.relu(self.BN(self.init_conv(inputs)))
        rb1 = self.pool(self.rb1(init_conv))
        rb2 = self.pool(self.rb2(rb1))
        rb3 = self.pool(self.rb3(rb2))
        rb4 = self.pool(self.rb4(rb3))
        rb5 = self.pool(self.rb5(rb4))
        rb6 = self.pool(self.rb6(rb5))
        feature_vec = torch.flatten(rb6, start_dim=1)
        return feature_vec


class Decoder(nn.Module):
    """
    Decoder class for segmentation, designed to upsample feature maps to 144x200.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.rb1 = ResBlock(333, 256, 3, 1, 1, 'decode')
        self.upconv0 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # Upsample to 4x6
        self.rb2 = ResBlock(128, 128, 3, 1, 1, 'decode')
        self.upconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # Upsample to 8x12
        self.rb3 = ResBlock(64, 64, 3, 1, 1, 'decode')
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # Upsample to 16x24
        self.rb4 = ResBlock(32, 32, 3, 1, 1, 'decode')
        self.upconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # Upsample to 32x48
        self.rb5 = ResBlock(16, 16, 3, 1, 1, 'decode')
        self.upconv4 = nn.ConvTranspose2d(16, 8, 4, 2, 1)  # Upsample to 64x96
        self.rb6 = ResBlock(8, 8, 3, 1, 1, 'decode')
        self.upconv5 = nn.ConvTranspose2d(8, 3, 4, 2, 1)  # Upsample to 128x192
        self.out_conv = nn.Conv2d(3, 3, 3, 1, 1)  # Adjust output channels to the number of classes

    def forward(self, feature_vec, inference=False):
        # Start by reshaping feature_vec to a shape that makes sense for decoding
        x = feature_vec.view(-1, 333, 2, 3)  # Adjust this if necessary
        x = self.rb1(x)
        x = self.upconv0(x)
        x = self.rb2(x)
        x = self.upconv1(x)  # Upsample to 8x12
        x = self.rb3(x)
        x = self.upconv2(x)  # Upsample to 16x24
        x = self.rb4(x)
        x = self.upconv3(x)  # Upsample to 32x48
        x = self.rb5(x)
        x = self.upconv4(x)  # Upsample to 64x96
        x = self.rb6(x)
        x = self.upconv5(x)  # Upsample to 128x192
        x = F.interpolate(x, size=[144, 200], mode='bilinear', align_corners=False)
        out = self.out_conv(x)  # Final output with the number of classes
        if inference:
            out = torch.argmax(out, dim=1)
        return out


class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """

    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p

    def forward(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# Training Loop
def train(model, dataloader, criterion, optimizer, **kwargs):
    num_epochs = kwargs.get('num_epochs', 100)
    scheduler = kwargs.get('scheduler', None)
    checkpoint_folder = kwargs.get('checkpoint_folder', None)
    patience = max_patience = 8
    best_loss = np.inf
    best_model = copy.deepcopy(model)
    for epoch in range(num_epochs):
        print('Processing epoch:', epoch + 1, 'lr:', optimizer.param_groups[0]['lr'], 'patience:', patience)
        model.train()
        running_loss = 0.0
        for i, (EMBEDDINGS, _, _, _, name) in enumerate(dataloader.trn_loader):
            images = EMBEDDINGS.to(device).unsqueeze(1) if EMBEDDINGS.dim() == 3 else EMBEDDINGS.to(device)
            targets = images.squeeze(1).long()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader.trn_loader):.4f}")
        val_loss = evaluate(model, dataloader, criterion)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
            patience = max_patience
        else:
            patience -= 1
        if patience <= max_patience - 3 and scheduler is not None:
            scheduler.step()
        if patience == 0:
            print("Early stopping")
            break
        if (epoch+1) % 15 == 0 and checkpoint_folder is not None:
            print(f"Saving model at epoch {epoch + 1} into {checkpoint_folder}")
            output_path = f'{checkpoint_folder}/best_model_epoch_{epoch + 1}_best_v_loss_{best_loss:.4f}.pth'
            torch.save(best_model.state_dict(), output_path)


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, (EMBEDDINGS, _, _, _, name) in enumerate(dataloader.val_loader):
            images = EMBEDDINGS.to(device).unsqueeze(1) if EMBEDDINGS.dim() == 3 else EMBEDDINGS.to(device)
            targets = images.squeeze(1).long()
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
        val_loss = running_loss / len(dataloader.val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        return val_loss


def parse_args():
    parser = argparse.ArgumentParser(description='U-Net Autoencoder Training')
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('-ga', '--gamma', type=float, default=0.9)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
    print(f"Number of parameters in the model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification (0, 1, 2)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=args.batch_size, pca=None,
                                                   try_cache=False)
    train(model, antenna_dataset_loader, criterion, optimizer, num_epochs=args.num_epochs, scheduler=scheduler,
          checkpoint_folder=args.checkpoint_path)

