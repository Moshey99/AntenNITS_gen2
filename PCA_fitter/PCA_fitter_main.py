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
import argparse
from sklearn.decomposition import PCA
import cv2

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
parser.add_argument('--image_size', type=tuple, nargs='+', default=(144, 200), help='Size of the Image')
parser.add_argument('--split_ratio', type=float, default=0.999, help='Ratio to split the dataset')
parser.add_argument('--n_comp', type=int, default=2000, help='Number of Components for PCA')
args = parser.parse_args()

image_size = args.image_size
data_path = args.data_path

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


def plot_explained_variance(pca: PCA):
    explained_variance = pca.explained_variance_ratio_
    explained_variance_cumulative = np.cumsum(explained_variance)
    plt.plot(explained_variance_cumulative)
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance')
    plt.show()


# DataLoader Function
class Imageprep(torch.utils.data.Dataset):
    def __init__(self, images):
        super().__init__()
        self.paths = images
        self.len = len(self.paths)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        path = self.paths[index]
        image = binarize(np.load(path))
        image = cv2.resize(image, (image_size[1], image_size[0]))
        image = torch.tensor(image).float().unsqueeze(0)
        return image


if __name__ == '__main__':
    print(args)
    train_set = Imageprep(train_imgs)

    pca_data_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set), shuffle=True)
    fit = True
    show_explained_variance = True
    print('Preparing data for fitting PCA...')
    for X in pca_data_loader:
        print('Data Shape: ', X.shape, '')
        X = X.view(X.size(0), -1)
        if fit:
            pca = PCA(n_components=args.n_comp)
            print('Fitting PCA...')
            pca.fit(X.detach().cpu().numpy())
            print('PCA Fitted!')
            pickle.dump(pca, open(os.path.join(data_path, 'pca_model.pkl'), 'wb'))
            print('PCA Model Saved in ', os.path.join(data_path, 'pca_model.pkl'))
            plot_explained_variance(pca) if show_explained_variance else None
        else:
            pca = pickle.load(open(os.path.join(data_path, 'pca_model.pkl'), 'rb'))
        for i in range(3):
            example_to_show = X[i:i + 1].detach().cpu().numpy()
            og_image = example_to_show.reshape(image_size[0], image_size[1])
            plt.imshow(og_image, cmap='gray')
            plt.title('Original Image')
            plt.figure()
            reconstructed_example = pca.inverse_transform(pca.transform(example_to_show))
            reconstructed_image = reconstructed_example.reshape(image_size[0], image_size[1])
            plt.imshow(binarize(reconstructed_image, 0.5, 1.5), cmap='gray')
            plt.title('Reconstructed Image')
            plt.show()
