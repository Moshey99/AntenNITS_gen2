import matplotlib.pyplot as plt

from models.forward_GammaRad import forward_GammaRad
from losses import GammaRad_loss
from AntennaDesign.utils import *

import argparse
import torch
import os
import pickle
from typing import Tuple, List


def plot_condition(condition: Tuple[torch.Tensor, torch.Tensor], freqs: np.ndarray) -> plt.Figure:
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    gamma, rad = condition
    gamma_amp, gamma_phase = gamma[:, :gamma.shape[1] // 2], gamma[:, gamma.shape[1] // 2:]
    ax1.plot(freqs, gamma_amp[0].cpu().detach().numpy(), 'b-')
    ax11 = ax1.twinx()
    ax11.plot(freqs, gamma_phase[0].cpu().detach().numpy(), 'r-')
    ax2.imshow(rad[0, 0].cpu().detach().numpy())
    ax3.imshow(rad[0, 2].cpu().detach().numpy())
    ax4.imshow(rad[0, 4].cpu().detach().numpy())
    ax1.set_title('gamma')
    ax1.set_ylabel('amplitude', color='b')
    ax1.set_ylim([-20, 0])
    ax11.set_ylim([-np.pi, np.pi])
    ax11.set_ylabel('phase', color='r')
    ax2.set_title('radiation f=2.1GHz')
    ax3.set_title('radiation f=2.4GHz')
    ax4.set_title('radiation f=2.7GHz')
    return fig


def produce_stats_all_dataset(gamma_stats: List[tuple], radiation_stats: List[tuple]):
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
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
    parser.add_argument('--rad_range', type=list, default=[-55, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=1e-3, help='controls the influence of geometry loss')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints\forward_best_dict.pth')
    return parser.parse_args()


if __name__ == "__main__":
    all_gamma_stats, all_radiation_stats = [], []
    plot_GT_vs_pred = True
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args, device)
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1)
    model = forward_GammaRad(radiation_channels=12)
    scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
    scaler_manager.try_loading_from_cache()
    if scaler_manager.scaler is None:
        raise ValueError('Scaler not found.')
    for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
        if idx == 1:
            break
        EMBEDDINGS, GAMMA, RADIATION, ENV, _ = sample
        embeddings, gamma, radiation, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
            scaler_manager.scaler.forward(ENV).to(device)
        geometry = torch.cat((embeddings, env), dim=1)
        target = (gamma, radiation)
        gamma_pred, rad_pred = model(geometry)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    loss_fn = GammaRad_loss(geo_weight=args.geo_weight)
    model.to(device)

    with torch.no_grad():
        model.eval()
        for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV, name = sample
            embeddings, gamma, radiation, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
                scaler_manager.scaler.forward(ENV).to(device)
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
