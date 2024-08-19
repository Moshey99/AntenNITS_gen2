import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.forward_GammaRad import forward_GammaRad

from AntennaDesign.utils import *
from forward_model_evaluate_main import plot_condition

from PCA_fitter.PCA_fitter_main import binarize


def sort_by_metric(*args):
    sorting_idxs = []
    for i, metric in enumerate(args):
        if i == len(args) - 1:
            metric = -metric  # reverse the msssim metric because we look for the minimum
        sorting_idxs.append(torch.argsort(metric, descending=False))
    sorting_idxs = torch.cat(sorting_idxs).reshape(len(args), -1)
    n_samples = sorting_idxs.shape[1]
    sample_score = torch.zeros(n_samples)
    for num_sample in range(n_samples):
        sample_locations = torch.argwhere(sorting_idxs == num_sample)[:, 1].to(float)
        sample_locations[-1] *= 1.5  # avg and max are correlated, so we give more weight msssim to balance it
        sample_score[num_sample] = sample_locations.mean()
    sort_idx = torch.argsort(sample_score)
    return sort_idx


def model_init_shape(model, data_loader: AntennaDataSetsLoader):
    with torch.no_grad():
        for idx, (EMBEDDINGS, _, _, ENV, _) in enumerate(data_loader.trn_loader):
            if idx == 0:
                embeddings, env = EMBEDDINGS.to(device), scaler_manager.scaler.forward(ENV).to(device)
                geometry = torch.cat((embeddings, env), dim=1)
                _, _ = model(geometry)
                break


def extend_to_fit_samples(num_samples: int, env: torch.Tensor, gamma: torch.Tensor, rad: torch.Tensor):
    gamma = torch.tile(gamma, (num_samples, 1))
    rad = torch.tile(rad, (num_samples, 1, 1, 1))
    env = torch.tile(env, (num_samples, 1))
    return gamma, rad, env


def image_from_embeddings(pca_, embeddings: np.ndarray, shape=(144, 200)):
    ant_resized = pca_.inverse_transform(embeddings).reshape(shape)
    return ant_resized


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
parser.add_argument('--forward_checkpoint_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints\forward_epoch300.pth')
parser.add_argument('-o', '--output_folder', type=str, default=None)
args = parser.parse_args()
device = torch.device("cpu")
data_path = args.data_path
output_folder = args.output_folder if args.output_folder is not None else os.path.join(args.data_path,
                                                                                       'checkpoints_inverse')

samples_folder = os.path.join(output_folder, 'samples')
samples_names = os.listdir(samples_folder)
pca = pickle.load(open(os.path.join(data_path, 'pca_model.pkl'), 'rb'))
antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=1, pca=pca, try_cache=False)
scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
scaler_manager.try_loading_from_cache()
if scaler_manager.scaler is None:
    raise ValueError('Scaler not found.')
model = forward_GammaRad(radiation_channels=12)
model_init_shape(model, antenna_dataset_loader)
model.load_state_dict(torch.load(args.forward_checkpoint_path, map_location=device))

with torch.no_grad():
    model.eval()
    for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
        if all([name[0] not in sample_name for sample_name in samples_names]) or int(name[0]) < 50000:
            continue
        print(f'evaluating samples for antenna {name[0]}.')
        x, gamma, rad, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
            scaler_manager.scaler.forward(ENV).to(device)

        samples = np.load(os.path.join(samples_folder, f'sample_{name[0]}.npy'))
        gamma_ext, rad_ext, env_ext = extend_to_fit_samples(samples.shape[0], env, gamma, rad)
        geometries = torch.cat((torch.tensor(samples).float().to(device), env_ext), dim=1)
        gamma_pred, rad_pred = model(geometries)
        gamma_pred_dB = gamma_to_dB(gamma_pred)
        gamma_stats = produce_gamma_stats(gamma_ext, gamma_pred_dB, dataset_type='dB')
        radiation_stats = produce_radiation_stats(rad_ext, rad_pred)
        sorting_idxs = sort_by_metric(*gamma_stats, *radiation_stats)
        gamma_pred_dB_sorted = gamma_pred_dB[sorting_idxs]
        rad_pred_sorted = rad_pred[sorting_idxs]
        embeddings_sorted = samples[sorting_idxs]
        plot_GT_vs_pred = True
        if plot_GT_vs_pred:
            fig_gt = plot_condition((gamma, rad), freqs=np.arange(gamma.shape[1] // 2))
            gamma_pred_dB_best = gamma_pred_dB_sorted[0].unsqueeze(0)
            rad_pred_best = rad_pred_sorted[0].unsqueeze(0)
            fig_pred = plot_condition((gamma_pred_dB_best, rad_pred_best), freqs=np.arange(gamma.shape[1] // 2))
            fig_gt.suptitle("ground truth")
            fig_pred.suptitle("generated antenna's prediction")
            plt.show()
            i = 0
            antenna_im = binarize(image_from_embeddings(pca, x.cpu().numpy().flatten()))
            plt.imshow(antenna_im)
            plt.title('Antenna')
            plt.figure()
            antenna_im = image_from_embeddings(pca, embeddings_sorted[i])
            plt.imshow(antenna_im)
            plt.title('Antenna generated, idx: ' + str(i))
            plt.show()
        pass
