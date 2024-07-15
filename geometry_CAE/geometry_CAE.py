# Import Libraries
import itertools
import os
import json
import random
import pickle
import torch
import torch.nn as nn
import torchvision
from PIL import Image
import glob
import matplotlib.pyplot as plt
import numpy as np
from resnet_vae import ResNet_VAE
import argparse
import pytorch_msssim
from sklearn.decomposition import PCA

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Argument Parser
parser = argparse.ArgumentParser(description='Convolutional AutoEncoder for Geometry Data')
parser.add_argument('--device', type=int, nargs='+', default=[2], help='Device to run the model on')
parser.add_argument('--lrs', type=float, nargs='+', default=[0.0005], help='Learning Rates to try')
parser.add_argument('--bs', type=int, nargs='+', default=[32], help='Batch Sizes to try')
parser.add_argument('--embed_sizes', type=int, nargs='+', default=[4096], help='Embedding Sizes to try')
parser.add_argument('--gamma', type=float, default=0.9, help='Gamma for StepLR')
parser.add_argument('--patiance', type=int, default=10, help='Patiance for Early Stopping')
parser.add_argument('--checkpoint_folder', type=str, default='checkpoints', help='Folder to save checkpoints')
parser.add_argument('--extra_string', type=str, default='', help='Extra string to add to the model name')
parser.add_argument('--split_ratio', type=float, default=0.9, help='Ratio to split the dataset')
parser.add_argument('--weight_mse', type=float, default=0.5, help='Weight for MSE Loss')
parser.add_argument('--images_folder', type=str, default='images', help='Folder to save images')
parser.add_argument('--kl_weight', type=float, default=0.01, help='Weight for KL Loss')
args = parser.parse_args()
print(args)


class mse_msssim_combined_mag_loss(nn.Module):
    def __init__(self, weight=0.5):
        super(mse_msssim_combined_mag_loss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM()
        assert 0 <= weight <= 1, 'Weight should be between 0 and 1'
        self.weight = weight

    def forward(self, pred, target):
        mse_loss = self.mse_loss(pred.view(pred.size(0), -1), target.view(target.size(0), -1))
        ssim_loss = -self.ssim_loss(pred, target)
        return self.weight * mse_loss + (1 - self.weight) * ssim_loss, mse_loss, ssim_loss


def loss_function(reconstruction_loss, target, pred, mu, logvar, kl_weight=0.01):
    recon_loss, mse, ssim = reconstruction_loss(pred, target)
    kl_loss = 0.5 * torch.mean(-1 - logvar + mu ** 2 + logvar.exp())
    return kl_weight * kl_loss + recon_loss, (kl_loss.item(), mse.item(), ssim.item())


# Load Config files
path = os.getcwd()
config_path = os.path.join(path, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

print("The Configuration Variables are:")
print('Configuration: ', config)

# Define Config variables
image_size = config['image_size']
data_path = config['DataPath']
batch_size = config['batch_size']
learning_rate = config['lr']
weight_decay = config['weight_decay']
epochs = config['n_epochs']
load_model = config['load_model']
embed_size = config['embedding_size']
embed_sizes = args.embed_sizes
lrs = args.lrs
bs_sizes = args.bs
print("\n____________________________________________________\n")
print("\nLoading Dataset into DataLoader...")

all_imgs = glob.glob(data_path + '/*.npy')
random.Random(42).shuffle(all_imgs)

# Train Images
split_ratio = args.split_ratio
split_index = int(len(all_imgs) * split_ratio)
train_imgs = all_imgs[:split_index]
test_imgs = all_imgs[split_index:]


def binarize(img, threshold=0.5):
    img[img < threshold] = 0
    img[img >= threshold] = 1
    return img


def standardize(X, avg=0.15446, std=0.35994):
    return (X - avg) / std


def evaluate(model, test_loader, loss_fn, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for X in test_loader:
            X = standardize(X)
            im = X.to(device)
            recon, mu, logvar, latents = model(im, latent_vec=True)
            loss, _ = loss_fn(mse_msssim_combined_mag_loss(weight=args.weight_mse), im, recon, mu, logvar)
            losses.append(loss.item())
    return np.mean(losses), np.std(losses)


# DataLoader Function
class imagePrep(torch.utils.data.Dataset):
    def __init__(self, images, transform, rotate=False, flip_horizontal=False, flip_vertical=False):
        super().__init__()
        self.paths = images
        self.len = len(self.paths)
        self.transform = transform
        self.rotate = rotate
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = (255 * np.load(path)[:, 0, :].T).astype(np.uint8)
        if self.rotate:
            image = image.T
        if self.flip_horizontal:
            image = np.flip(image, axis=1)
        if self.flip_vertical:
            image = np.flip(image, axis=0)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image


# Dataset Transformation Function
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                                    torchvision.transforms.Resize((image_size[0], image_size[1])),
                                                    torchvision.transforms.ToTensor()])

# Apply Transformations to Data
train_set = imagePrep(train_imgs, dataset_transform)

pca_data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
for X in pca_data_loader:
    print('Data Shape: ', X.shape, '')
    X = X.view(X.size(0), -1)
    pca = PCA(n_components=3000)
    print('Fitting PCA...')
    pca.fit(X.detach().cpu().numpy())
    pickle.dump(pca, open('pca_model.pkl', 'wb'))
    print('PCA Fitted!')
    for i in range(3):
        example_to_show = X[i:i+1].detach().cpu().numpy()
        og_image = example_to_show.reshape(image_size[0], image_size[1])
        plt.imshow(og_image, cmap='gray')
        plt.title('Original Image')
        plt.figure()
        reconstructed_example = pca.inverse_transform(pca.transform(example_to_show))
        reconstructed_image = reconstructed_example.reshape(image_size[0], image_size[1])
        plt.imshow(binarize(reconstructed_image), cmap='gray')
        plt.title('Reconstructed Image')
        plt.show()

    break
print("\nDataLoader Set!")
print("\n____________________________________________________\n")

print("\nBuilding Convolutional AutoEncoder Network Model...")

print("\n____________________________________________________\n")

model_loss_dicts = {}
model_weights_dicts = {}
for learning_rate, batch_size, embed_size in itertools.product(lrs, bs_sizes, embed_sizes):
    patiance = args.patiance
    print(f"Learning Rate: {learning_rate} | Batch Size: {batch_size} | Embedding Size: {embed_size}")
    model_name = f'geometry_convVAE_model_lr{learning_rate}_bs{batch_size}_ems{embed_size}_exs{args.extra_string}.pth'
    print("\n____________________________________________________\n")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(imagePrep(test_imgs, dataset_transform), batch_size=batch_size)

    # defining the device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device[0]}')
        convAE_model = torch.nn.DataParallel(ResNet_VAE(CNN_embed_dim=embed_size), device_ids=args.device).to(device)
    else:
        device = torch.device("cpu")
        convAE_model = ResNet_VAE(CNN_embed_dim=embed_size).to(device)
    if load_model:
        convAE_model = torch.nn.DataParallel(ResNet_VAE(CNN_embed_dim=embed_size), device_ids=args.device).to(device)
        convAE_model.load_state_dict(
            torch.load('geometry_convVAE_model_lr0.0005_bs32_ems4096_exs_kl0.001_mse1_epo5.pth', map_location=device))
    optimizer = torch.optim.Adam(convAE_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("\nTraining the Convolutional AutoEncoder Model on Training Data...")

    # Training of Model

    losses = []
    best_loss = 999999
    convAE_model.train()
    for epoch in range(epochs):
        print('Epoch: {} | Patiance: {}'.format(epoch, patiance))
        print('Training...')
        train_loss = 0
        debug_losses = np.array([0, 0, 0], dtype=np.float64)
        for i, X in enumerate(train_loader):
            X = standardize(X)
            img = X.to(device)
            recon, mu, logvar, latents = convAE_model(img, latent_vec=True)
            if args.images_folder is not None and i == 0:
                original_img = img.detach().cpu().numpy()[1][0]
                reconstructed_img = recon.detach().cpu().numpy()[1][0]
                plt.figure()
                plt.imshow(np.hstack((original_img, reconstructed_img)), cmap='gray')
                plt.title('Original Image | Reconstructed Image')
                image_path = os.path.join(args.images_folder, model_name.replace('.pth', f'_epoch_{epoch}_image.png'))
                plt.savefig(image_path)
                plt.clf()
                print(f"Image Saved at {image_path}")
            # plt.imshow(recon.cpu().data[1][0], cmap='gray')
            # plt.title('Reconstructed Image')
            # plt.figure()
            # plt.imshow(img.cpu().data[1][0], cmap='gray')
            # plt.title('Original Image')
            # plt.show()
            rec_loss = mse_msssim_combined_mag_loss(weight=args.weight_mse)
            loss, (kl, mse, ssim) = loss_function(rec_loss, target=img, pred=recon, mu=mu, logvar=logvar,
                                                  kl_weight=args.kl_weight)

            # Backward Propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            debug_losses += np.array([kl, mse, ssim])
        print('Evaluating...')
        test_loss, test_loss_std = evaluate(model=convAE_model, test_loader=test_loader, loss_fn=loss_function,
                                            device=device)
        train_loss = train_loss / len(train_loader)
        debug_losses = np.round(debug_losses / len(train_loader), 4)
        losses.append(train_loss)
        if test_loss < best_loss:
            patiance = args.patiance
            best_loss = test_loss
            best_weights = convAE_model.state_dict()
        else:
            patiance -= 1
            scheduler.step()
            if patiance == 0:
                print(f"Early Stopping at Epoch: {epoch}")
                break
        print(f"KL Loss: {debug_losses[0]} | MSE Loss: {debug_losses[1]} | SSIM Loss: {debug_losses[2]}")
        print("Train Loss: {:.4f}".format(train_loss),
              f" | Test Loss: {np.round(test_loss, 4)} +- {np.round(test_loss_std, 4)}, Best Loss: {np.round(best_loss, 4)}\n")
        if (epoch + 5) % 10 == 0 and args.checkpoint_folder is not None:
            checkpoint_dir = os.path.join(args.checkpoint_folder, model_name.replace('.pth', f'_epo{epoch}.pth'))
            torch.save(convAE_model.state_dict(), checkpoint_dir)
            print(f"Model Saved at Epoch {epoch} in {checkpoint_dir}")
    model_loss_dicts[model_name] = best_loss.item()
    model_weights_dicts[model_name] = best_weights
    torch.save(best_weights, model_name)

    print("\n____________________________________________________\n")
print(model_loss_dicts)
best_model = min(model_loss_dicts, key=model_loss_dicts.get)
print(f"Best Model: {best_model} | Best Loss: {model_loss_dicts[best_model]}")
