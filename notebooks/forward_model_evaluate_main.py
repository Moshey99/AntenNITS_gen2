import matplotlib.pyplot as plt

from models.forward_GammaRad import forward_GammaRad
from losses import GammaRad_loss
from AntennaDesign.utils import *

import argparse
import torch
import os
import pickle


def plot_condition(condition, freqs, to_dB=False):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    gamma, rad = condition
    gamma_amp, gamma_phase = gamma[:, :gamma.shape[1] // 2], gamma[:, gamma.shape[1] // 2:]
    if to_dB:
        gamma_amp = 10 * torch.log10(gamma_amp)
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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
    parser.add_argument('--forward_model_path_gamma', type=str,
                        default=r'checkpoints/forward_gamma_smoothness_0.001_0.0001.pth')
    parser.add_argument('--forward_model_path_radiation', type=str,
                        default=r'checkpoints/forward_radiation_huberloss.pth')
    parser.add_argument('--rad_range', type=list, default=[-55, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=1e-3, help='controls the influence of geometry loss')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints\forward_12radchannels.pth')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    args.checkpoint_path = args.checkpoint_path.replace('forward_12radchannels.pth', 'forward_epoch130.pth')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args, device)
    pca = pickle.load(open(os.path.join(args.data_path, 'pca_model.pkl'), 'rb'))
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1, pca=pca, try_cache=True)
    model = forward_GammaRad(radiation_channels=12)
    scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
    scaler_manager.try_loading_from_cache()
    if scaler_manager.scaler is None:
        raise ValueError('Scaler not found.')
    for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
        if idx == 1:
            break
        EMBEDDINGS, GAMMA, RADIATION, ENV = sample
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
            EMBEDDINGS, GAMMA, RADIATION, ENV = sample
            embeddings, gamma, radiation, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
                scaler_manager.scaler.forward(ENV).to(device)
            if gamma.min() > -3.2:
                continue
            print(idx)
            plot_condition((GAMMA, RADIATION), freqs=np.arange(GAMMA.shape[1]//2))
            geometry = torch.cat((embeddings, env), dim=1)
            target = (gamma, radiation)
            gamma_pred, rad_pred = model(geometry)
            plot_condition((gamma_pred, rad_pred), freqs=np.arange(gamma_pred.shape[1]//2), to_dB=True)
            plt.show()
            output = (gamma_pred, rad_pred, geometry)
    with torch.no_grad():
        model.eval()
        val_loss = 0
        for idx, sample in enumerate(antenna_dataset_loader.val_loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV = sample
            embeddings, gamma, radiation, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
                scaler_manager.scaler.forward(ENV).to(device)
            geometry = torch.cat((embeddings, env), dim=1)
            target = (gamma, radiation)
            gamma_pred, rad_pred = model(geometry)

    torch.save(best_model.state_dict(), args.checkpoint_path.replace('.pth', '_best_dict.pth'))
    torch.save(best_model, args.checkpoint_path.replace('.pth', '_best_instance.pth'))
    print('Training finished.')
    print(f'Best loss: {best_loss}')
    print(f'Best model saved at: {args.checkpoint_path}')
