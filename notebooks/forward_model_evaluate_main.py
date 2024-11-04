import matplotlib.pyplot as plt

from models.forward_GammaRad import forward_GammaRad
from losses import GammaRad_loss
from AntennaDesign.utils import *

import argparse
import torch
import os
import pickle
from typing import Tuple, List, Union


def plot_radiation_pattern(radiation_db, ax):
    """
    Plots a 3D radiation pattern on a given axis.

    Parameters:
    - radiation_db: 2D numpy array with dB values
    - ax: The axis on which to plot the 3D radiation pattern
    """
    # Generate theta and phi grids
    num_theta, num_phi = radiation_db.shape
    theta = np.linspace(0, np.pi, num_theta)  # theta goes from 0 to pi
    phi = np.linspace(0, 2 * np.pi, num_phi)  # phi goes from 0 to 2*pi

    # Create a meshgrid of theta and phi
    phi, theta = np.meshgrid(phi, theta)

    # Convert dB to linear scale
    r = 10 ** (radiation_db / 10)

    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Plot the 3D radiation pattern on the provided axis
    ax.plot_surface(x, y, z, facecolors=plt.cm.jet(radiation_db), rstride=1, cstride=1, alpha=0.8)


def plot_condition(condition: Tuple[torch.Tensor, torch.Tensor], freqs: np.ndarray, plot_type: str = '2d') -> plt.Figure:
    gamma, rad = condition
    gamma_amp, gamma_phase = gamma[:, :gamma.shape[1] // 2], gamma[:, gamma.shape[1] // 2:]

    # Create the subplots, ensuring 3D projection for 3D plots
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(141)  # 2D plot for gamma
    ax2 = fig.add_subplot(142, projection='3d' if plot_type == '3d' else None)
    ax3 = fig.add_subplot(143, projection='3d' if plot_type == '3d' else None)
    ax4 = fig.add_subplot(144, projection='3d' if plot_type == '3d' else None)

    # Plot gamma amplitude and phase on ax1
    ax1.plot(freqs, gamma_amp[0].cpu().detach().numpy(), 'b-')
    ax11 = ax1.twinx()
    ax11.plot(freqs, gamma_phase[0].cpu().detach().numpy(), 'r-')
    ax1.set_title('gamma')
    ax1.set_ylabel('amplitude', color='b')
    ax1.set_ylim([-20, 0])
    ax11.set_ylim([-np.pi, np.pi])
    ax11.set_ylabel('phase', color='r')
    rad_first_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(rad[0, 0])**2 + radiation_mag_to_linear(rad[0, 1])**2))
    rad_second_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(rad[0, 2])**2 + radiation_mag_to_linear(rad[0, 3])**2))
    rad_third_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(rad[0, 4])**2 + radiation_mag_to_linear(rad[0, 5])**2))
    # Plot radiation patterns
    if plot_type == '2d':
        # 2D plots using imshow
        ax2.imshow(rad_first_freq.cpu().detach().numpy(), vmin=-20, vmax=5, cmap='jet')
        ax3.imshow(rad_second_freq.cpu().detach().numpy(), vmin=-20, vmax=5, cmap='jet')
        ax4.imshow(rad_third_freq.cpu().detach().numpy(), vmin=-20, vmax=5, cmap='jet')
    elif plot_type == '3d':
        # 3D plots using plot_radiation_pattern
        plot_radiation_pattern(rad_first_freq.cpu().detach().numpy(), ax2)
        plot_radiation_pattern(rad_second_freq.cpu().detach().numpy(), ax3)
        plot_radiation_pattern(rad_third_freq.cpu().detach().numpy(), ax4)

    # Set titles for the radiation pattern subplots
    ax2.set_title('rad f=1.5GHz')
    ax3.set_title('rad f=2.1GHz')
    ax4.set_title('rad f=2.4GHz')

    return fig


def produce_stats_all_dataset(gamma_stats: Union[List[Tuple], np.ndarray], radiation_stats: Union[List[Tuple], np.ndarray]):
    print('--' * 20)
    gamma_stats_gathered = torch.tensor(gamma_stats)
    gamma_stats_mean = torch.nanmean(gamma_stats_gathered, dim=0).numpy()
    assert len(gamma_stats_mean) == 4, 'gamma stats mean should have 4 elements' \
                                       ' (avg mag, max mag, avg phase, max phase)'
    metrics_gamma_keys = [x + ' diff' for x in ['avg mag', 'max mag', 'avg phase', 'max phase']]
    stats_dict_gamma = dict(zip(metrics_gamma_keys, gamma_stats_mean))
    print(f'GAMMA STATS, averaged over entire dataset: {stats_dict_gamma}')
    radiation_stats_gathered = torch.tensor(radiation_stats)
    radiation_stats_mean = torch.nanmean(radiation_stats_gathered, dim=0).numpy()
    assert len(radiation_stats_mean) == 4, 'radiation stats mean should have 4 elements' \
                                           ' (avg mag, max mag, avg phase, msssim)'
    metrics_rad_keys = [x + ' diff' for x in ['avg mag', 'max mag', 'avg phase', 'msssim']]
    stats_dict_rad = dict(zip(metrics_rad_keys, radiation_stats_mean))
    print(f'RADIATION STATS, averaged over entire dataset: {stats_dict_rad}')
    print('--' * 20)
    pass


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_110k_150k_processed')
    parser.add_argument('--rad_range', type=list, default=[-20, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=0., help='controls the influence of geometry loss')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_110k_150k_processed\checkpoints\forward_best_dict.pth")
    return parser.parse_args()


if __name__ == "__main__":
    all_gamma_stats, all_radiation_stats = [], []
    plot_GT_vs_pred = False
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args, device)
    # pca = pickle.load(open(os.path.join(args.data_path, 'pca_model.pkl'), 'rb'))
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1, try_cache=True)
    model = forward_GammaRad(radiation_channels=12)
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'ant_scaler.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
        if idx == 1:
            break
        EMBEDDINGS, GAMMA, RADIATION, ENV, _ = sample
        embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
            GAMMA.to(device), RADIATION.to(device), \
            env_scaler_manager.scaler.forward(ENV).float().to(device)
        geometry = torch.cat((embeddings, env), dim=1)
        target = (gamma, radiation)
        gamma_pred, rad_pred = model(geometry)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    loss_fn = GammaRad_loss(geo_weight=args.geo_weight)
    model.to(device)

    with torch.no_grad():
        model.eval()
        for idx, sample in enumerate(antenna_dataset_loader.val_loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV, name = sample
            embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            if antenna_dataset_loader.batch_size == 1 and gamma[:, :int(gamma.shape[1] // 2)].min() > -1.5:
                print(f'Antenna #{name[0]} has bad resonance, skipping.')
                continue  # skip antennas without good resonances (if batch size is 1, i.e. that's the only one in gamma)
            print(f'Working on antenna #{name[0]}')
            geometry = torch.cat((embeddings, env), dim=1)
            target = (gamma, radiation)
            gamma_pred, rad_pred = model(geometry)
            gamma_pred_dB = gamma_to_dB(gamma_pred)
            gamma_stats = produce_gamma_stats(gamma, gamma_pred_dB, dataset_type='dB')
            all_gamma_stats.append(gamma_stats)
            radiation_stats = produce_radiation_stats(radiation, rad_pred)
            all_radiation_stats.append(radiation_stats)
            if plot_GT_vs_pred:
                plot_condition((GAMMA, RADIATION), freqs=np.arange(GAMMA.shape[1] // 2))
                plot_condition((gamma_pred_dB, rad_pred), freqs=np.arange(gamma_pred_dB.shape[1] // 2))
                plt.show()
        produce_stats_all_dataset(all_gamma_stats, all_radiation_stats)
