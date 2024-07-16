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
from sklearn.decomposition import PCA
import cv2

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

all_imgs = glob.glob(os.path.join(data_path, '*', 'antenna.npy'))
random.Random(42).shuffle(all_imgs)

# Train Images
split_ratio = args.split_ratio
split_index = int(len(all_imgs) * split_ratio)
train_imgs = all_imgs[:split_index]
test_imgs = all_imgs[split_index:]


def binarize(img, nonmetal_threshold=0.5, feed_threshold=1.5):
    img[img < nonmetal_threshold] = 0
    img[img >= feed_threshold] = 2
    img[(img >= nonmetal_threshold) & (img < feed_threshold)] = 1
    return img


# DataLoader Function
class imagePrep(torch.utils.data.Dataset):
    def __init__(self, images, rotate=False, flip_horizontal=False, flip_vertical=False):
        super().__init__()
        self.paths = images
        self.len = len(self.paths)
        self.rotate = rotate
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = np.load(path)
        if self.rotate:
            image = image.T
        if self.flip_horizontal:
            image = np.flip(image, axis=1)
        if self.flip_vertical:
            image = np.flip(image, axis=0)
        image = cv2.resize(image, (image_size[1], image_size[0]))
        image = torch.tensor(image).float().unsqueeze(0)
        return image




# Apply Transformations to Data
train_set = imagePrep(train_imgs)

pca_data_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
fit = False
for X in pca_data_loader:
    print('Data Shape: ', X.shape, '')
    X = X.view(X.size(0), -1)
    if fit:
        pca = PCA(n_components=2000)
        print('Fitting PCA...')
        pca.fit(X.detach().cpu().numpy())
        print('PCA Fitted!')
        explained_variance = pca.explained_variance_ratio_
        pickle.dump(pca, open(os.path.join(data_path, 'pca_model.pkl'), 'wb'))
    else:
        pca = pickle.load(open(os.path.join(data_path, 'pca_model.pkl'), 'rb'))
    for i in range(6):
        example_to_show = X[i:i+1].detach().cpu().numpy()
        og_image = example_to_show.reshape(image_size[0], image_size[1])
        plt.imshow(og_image, cmap='gray')
        plt.title('Original Image')
        plt.figure()
        reconstructed_example = pca.inverse_transform(pca.transform(example_to_show))
        reconstructed_image = reconstructed_example.reshape(image_size[0], image_size[1])
        plt.imshow(binarize(reconstructed_image,0.5,1.5), cmap='gray')
        plt.title('Reconstructed Image')
        plt.show()
