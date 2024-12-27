import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt

from models.forward_GammaRad import forward_GammaRad
from AntennaDesign.utils import *

import argparse
import torch
from typing import Tuple


def plot_condition(condition: Tuple[torch.Tensor, torch.Tensor],
                   freqs: np.ndarray = np.arange(start=300, stop=3000.1, step=10.8),
                   plot_type: str = '2d',
                   title: str = '') -> plt.Figure:
    gamma, rad = condition
    gamma_sep = gamma.shape[1] // 2
    gamma_amp, gamma_phase = gamma[:, :gamma_sep], gamma[:, gamma_sep:]

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
    ax1.set_xlabel('frequency [MHz]')
    rad_first_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(rad[0, 0])**2 + radiation_mag_to_linear(rad[0, 1])**2))
    rad_second_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(rad[0, 2])**2 + radiation_mag_to_linear(rad[0, 3])**2))
    rad_third_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(rad[0, 4])**2 + radiation_mag_to_linear(rad[0, 5])**2))
    # Plot radiation patterns
    if plot_type == '2d':
        # 2D plots using imshow
        ax2.imshow(rad_first_freq.cpu().detach().numpy(), vmin=-10, vmax=5, cmap='jet')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.imshow(rad_second_freq.cpu().detach().numpy(), vmin=-10, vmax=5, cmap='jet')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.imshow(rad_third_freq.cpu().detach().numpy(), vmin=-10, vmax=5, cmap='jet')
        ax4.set_xticks([])
        ax4.set_yticks([])

    # Set titles for the radiation pattern subplots
    ax2.set_title('rad f=1.5GHz')
    ax3.set_title('rad f=2.1GHz')
    ax4.set_title('rad f=2.4GHz')
    fig.suptitle(title)
    return fig


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('--rad_range', type=list, default=[-15, 5], help='range of radiation values for scaling')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints\updated_forward_best_dict.pth")
    parser.add_argument('--repr_mode', type=str, help='use relative repr. for ant and env', default='abs')
    return parser.parse_args()


if __name__ == "__main__":
    all_gamma_stats, all_radiation_stats = [], []
    plot_GT_vs_pred = True
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args, device)
    # pca = pickle.load(open(os.path.join(args.data_path, 'pca_model.pkl'), 'rb'))
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1)
    model = forward_GammaRad(radiation_channels=12)
    scaler_name = 'scaler' if args.repr_mode == 'abs' else 'scaler_rel'
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
        EMBEDDINGS, GAMMA, RADIATION, ENV, _ = sample
        embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
            GAMMA.to(device), RADIATION.to(device), \
            env_scaler_manager.scaler.forward(ENV).float().to(device)
        geometry = torch.cat((embeddings, env), dim=1)
        target = (gamma, radiation)
        gamma_pred, rad_pred = model(geometry)
        break
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)

    with torch.no_grad():
        model.eval()
        for idx, sample in enumerate(antenna_dataset_loader.val_loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV, name = sample
            embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            gamma_mag = gamma[:int(gamma.shape[1] // 2)]
            if gamma_mag.min() > -5:
                print(f'Antenna #{name[0]} has bad resonance, skipping.')
                continue
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
                plot_condition((GAMMA, RADIATION))
                plot_condition((gamma_pred_dB, rad_pred))
                plt.show()
        produce_stats_all_dataset(all_gamma_stats, all_radiation_stats)
