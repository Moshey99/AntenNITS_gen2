from sklearn.neighbors import NearestNeighbors
import argparse
import torch
import numpy as np
from AntennaDesign.utils import *
import os
from forward_model_evaluate_main import plot_condition
from inverse_model_evaluate_main import arg_parser, sort_by_metric

if __name__ == "__main__":
    args = arg_parser().parse_args()
    n_neighbors = 100
    inverse_checkpoint_folder = os.path.join(args.data_path, 'checkpoints_inverse')
    output_folder_name = args.output_folder_name if args.output_folder_name is not None else 'NN_generated_antennas'
    output_folder = os.path.join(inverse_checkpoint_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cpu")
    data_path = args.data_path
    env_scaler_manager = ScalerManager(path=os.path.join(data_path, 'env_scaler.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(data_path, 'ant_scaler.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=24000, try_cache=False)
    antenna_dataset_loader.load_test_data(args.test_path) if args.test_path is not None else None
    path = args.test_path if args.test_path is not None else data_path
    loader = antenna_dataset_loader.tst_loader if args.test_path is not None else antenna_dataset_loader.val_loader
    with torch.no_grad():
        all_gamma_stats, all_radiation_stats = [], []
        plot_GT_vs_pred = False
        for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
            x_trn, gamma_trn, rad_trn, env_trn = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), ENV.to(device)
            env_trn_og_repr = env_to_dict_representation(env_trn)
            env_abs_trn_list = [list(env_trn_og_repr[i].values()) for i in range(env_trn.shape[0])]
            env_abs_trn = torch.tensor(env_abs_trn_list, device=device)
            break

        for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(loader):
            x_val, gamma_val, rad_val, env_val = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), ENV.to(device)
            envs_val_og_repr = env_to_dict_representation(env_val)
            env_abs_val_list = [list(envs_val_og_repr[i].values()) for i in range(env_val.shape[0])]
            envs_abs_val = torch.tensor(env_abs_val_list, device=device)
            break

        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(env_abs_trn)
        distances, nn_indices = nbrs.kneighbors(envs_abs_val)
        distances, nn_indices = distances[:, 3:], nn_indices[:, 3:]
        gamma_pred_dB = gamma_trn[nn_indices].squeeze()
        rad_pred = rad_trn[nn_indices].squeeze()
        ant_pred = x_trn[nn_indices].squeeze()
        all_gamma_stats_best, all_radiation_stats_best = [], []
        for sample_idx in range(nn_indices.shape[0]):
            sample_name = name[sample_idx]
            env_og_rel_repr = env_to_dict_representation(
                torch.tensor(np.load(os.path.join(path, sample_name, 'environment.npy'))[np.newaxis]))[0]

            gamma_stats = produce_gamma_stats(gamma_val[sample_idx].unsqueeze(0), gamma_pred_dB[sample_idx], dataset_type='dB', to_print=False)
            radiation_stats = produce_radiation_stats(rad_val[sample_idx].unsqueeze(0), rad_pred[sample_idx], to_print=False)
            sorting_idxs = sort_by_metric(*gamma_stats, *radiation_stats)
            found_valid_ant = False
            k = 0
            while k < len(sorting_idxs) and not found_valid_ant:
                best_nbr_index = sorting_idxs[k]
                gamma_stats_best = [gamma_stats[i][best_nbr_index] for i in range(len(gamma_stats))]
                radiation_stats_best = [radiation_stats[i][best_nbr_index] for i in range(len(radiation_stats))]
                sample_nn_indices = nn_indices[sample_idx]
                ant_neighbor = x_trn[sample_nn_indices[best_nbr_index]]
                ant_neighbor_inverse = ant_scaler_manager.scaler.inverse(ant_neighbor.unsqueeze(0))
                ant_neighbor_og_repr = ant_to_dict_representation(ant_neighbor_inverse)[0]
                if args.repr_mode == 'abs':
                    # making sure the representation is relative
                    ant_neighbor_og_repr = ant_abs2rel(ant_neighbor_og_repr, env_og_rel_repr)
                found_valid_ant = check_ant_validity(ant_neighbor_og_repr, env_og_rel_repr) == 1
                k += 1
            if k != len(sorting_idxs):
                print(f'Exited after {k} iterations for {sample_name}')
                with open(os.path.join(output_folder, f'ant_{sample_name}_nn.pickle'), 'wb') as ant_handle:
                    pickle.dump(ant_neighbor_og_repr, ant_handle)
                with open(os.path.join(output_folder, f'env_{sample_name}.pickle'), 'wb') as env_handle:
                    pickle.dump(env_og_rel_repr, env_handle)
                all_gamma_stats_best.append(gamma_stats_best)
                all_radiation_stats_best.append(radiation_stats_best)
            else:
                print(f'No valid antenna found for {sample_name} !!!!')
        print(f'For n_neighbors={n_neighbors}:')
        produce_stats_all_dataset(np.array(all_gamma_stats_best), np.array(all_radiation_stats_best))
