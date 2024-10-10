import os
from io import BytesIO
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from models.forward_GammaRad import forward_GammaRad

from AntennaDesign.utils import *
from forward_model_evaluate_main import plot_condition

from PCA_fitter.PCA_fitter_main import binarize
def figure_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = plt.imread(buf)
    buf.close()
    return img

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


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
parser.add_argument('--forward_checkpoint_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints\forward_epoch170.pth')
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
        if all([name[0] not in sample_name for sample_name in samples_names]):
            continue
        print(f'evaluating samples for antenna {name[0]}.')
        x, gamma, rad, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
            scaler_manager.scaler.forward(ENV).to(device)

        samples = np.load(os.path.join(samples_folder, f'sample_{name[0]}.npy'))
        samples = torch.tensor(
            antenna_dataset_loader.pca_wrapper.apply_binarization_on_components(samples)).float().to(device)
        gamma_ext, rad_ext, env_ext = extend_to_fit_samples(samples.shape[0], env, gamma, rad)
        geometries = torch.cat((samples, env_ext), dim=1)
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
            figs_gt, axs_gt = plt.subplots(2, 1, figsize=(5, 10))
            fig_gt = plot_condition((gamma, rad), freqs=np.arange(gamma.shape[1] // 2))
            img_gt = figure_to_image(fig_gt)
            plt.close(fig_gt)
            axs_gt[1].imshow(img_gt)
            axs_gt[1].set_title('GT target')
            axs_gt[1].axis('off')
            antenna_im = antenna_dataset_loader.pca_wrapper.image_from_components(x.cpu().numpy()).squeeze()
            axs_gt[0].imshow(antenna_im, vmin=0, vmax=2)
            axs_gt[0].set_title('GT Antenna')
            axs_gt[0].axis('off')

            fig, axs = plt.subplots(2, 3, figsize=(15, 10))

            for i in [0, 1, 2]:
                gamma_pred_dB_best = gamma_pred_dB_sorted[i].unsqueeze(0)
                rad_pred_best = rad_pred_sorted[i].unsqueeze(0)

                fig_pred = plot_condition((gamma_pred_dB_best, rad_pred_best), freqs=np.arange(gamma.shape[1] // 2))
                img_pred = figure_to_image(fig_pred)
                plt.close(fig_pred)
                axs[1, i].imshow(img_pred)
                axs[1, i].set_title(f'Generated Prediction, idx: {i}')
                axs[1, i].axis('off')  # Turn off axes for better visualization
                antenna_im = antenna_dataset_loader.pca_wrapper.image_from_components(
                    embeddings_sorted[i:i + 1]).squeeze()
                axs[0, i].imshow(antenna_im, vmin=0, vmax=2)
                axs[0, i].set_title(f' Generated Antenna, idx: {i}')
                axs[0, i].axis('off')

            plt.tight_layout()
            plt.show()
        pass
