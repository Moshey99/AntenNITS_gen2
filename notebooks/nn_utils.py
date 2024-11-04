from sklearn.neighbors import NearestNeighbors
import argparse
import torch
import numpy as np
from AntennaDesign.utils import *
import os
from models.forward_GammaRad import forward_GammaRad
from forward_model_evaluate_main import plot_condition, produce_stats_all_dataset
from inverse_model_evaluate_main import model_init_shape, arg_parser, ant_to_dict_representation, env_to_dict_representation

if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = torch.device("cpu")
    data_path = args.data_path
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'ant_scaler.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    model = forward_GammaRad(radiation_channels=12)
    antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=6949, try_cache=False)
    model_init_shape(model, antenna_dataset_loader)
    model.load_state_dict(torch.load(args.forward_checkpoint_path, map_location=device))
    with torch.no_grad():
        all_gamma_stats, all_radiation_stats = [], []
        plot_GT_vs_pred = False
        model.eval()
        for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
            x_trn, gamma_trn, rad_trn, env_trn = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            geometry_trn = torch.cat((x_trn, env_trn), dim=1)
            if idx == 1:
                break

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(geometry_trn)

        for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.val_loader):
            # if GAMMA[:, :GAMMA.shape[1] // 2].min() > -5:
            #     print(f'skipping antenna {name[0]}')
            #     continue
            print(f'evaluating samples for antenna {name[0]}.')
            x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            geometry_val = torch.cat((x, env), dim=1)
            distances, indices = nbrs.kneighbors(geometry_val)
            break


        gamma_pred_dB = gamma_trn[indices].squeeze()
        rad_pred = rad_trn[indices].squeeze()
        ant_pred = x_trn[indices].squeeze()
        env_pred = env_trn[indices].squeeze()


        gamma_stats = produce_gamma_stats(gamma, gamma_pred_dB, dataset_type='dB')
        radiation_stats = produce_radiation_stats(rad, rad_pred)

        all_gamma_stats = np.array(gamma_stats).T
        all_radiation_stats = np.array(radiation_stats).T
        produce_stats_all_dataset(all_gamma_stats, all_radiation_stats)
        #
        # env_og_repr = env_to_dict_representation(env_scaler_manager.scaler.inverse(env))[0]
        # gt_ant_og_repr = ant_to_dict_representation(ant_scaler_manager.scaler.inverse(x))[0]
        # pred_ant_og_repr = ant_to_dict_representation(ant_scaler_manager.scaler.inverse(ant_pred))
        # pred_env_og_repr = env_to_dict_representation(env_scaler_manager.scaler.inverse(env_pred))
        #
        # if plot_GT_vs_pred:
        #     figs_gt, axs_gt = plt.subplots(2, 1, figsize=(5, 10))
        #     fig_gt = plot_condition((gamma, rad), freqs=np.arange(gamma.shape[1] // 2), plot_type='3d')
        #     img_gt = figure_to_image(fig_gt)
        #     plt.close(fig_gt)
        #     axs_gt[1].imshow(img_gt)
        #     axs_gt[1].set_title('GT target')
        #     axs_gt[1].axis('off')
        #     antenna_fig = plot_antenna_figure(ant_parameters=gt_ant_og_repr, model_parameters=env_og_repr)
        #     antenna_im = figure_to_image(antenna_fig)
        #     plt.close(antenna_fig)
        #     axs_gt[0].imshow(antenna_im)
        #     axs_gt[0].set_title('GT Antenna')
        #     axs_gt[0].axis('off')
        #
        #     fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        #
        #     for i in [0, 1, 2]:
        #         gamma_pred_dB_best = gamma_pred_dB[i].unsqueeze(0)
        #         rad_pred_best = rad_pred[i].unsqueeze(0)
        #
        #         fig_pred = plot_condition((gamma_pred_dB_best, rad_pred_best), freqs=np.arange(gamma.shape[1] // 2),
        #                                   plot_type='3d')
        #         img_pred = figure_to_image(fig_pred)
        #         plt.close(fig_pred)
        #         axs[1, i].imshow(img_pred)
        #         axs[1, i].set_title(f'Generated Prediction, idx: {i}')
        #         axs[1, i].axis('off')  # Turn off axes for better visualization
        #         antenna_fig = plot_antenna_figure(ant_parameters=pred_ant_og_repr[i],
        #                                           model_parameters=env_og_repr)
        #         antenna_im = figure_to_image(antenna_fig)
        #         plt.close(antenna_fig)
        #         axs[0, i].imshow(antenna_im)
        #         axs[0, i].set_title(f' Generated Antenna, idx: {i}')
        #         axs[0, i].axis('off')
        #
        #     plt.tight_layout()
        #     plt.show()

