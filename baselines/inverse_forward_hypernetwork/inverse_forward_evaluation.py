import copy
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AntennaDesign.models.forward_GammaRad import forward_GammaRad
from AntennaDesign.models.inverse_hypernet import inverse_forward_concat
from nits.antenna_condition import GammaRadHyperEnv
from AntennaDesign.utils import *

import argparse
import torch
import pickle


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints_inverse\inv_forward_best_dict_bestloss_1.5098071080933284_lr_0.0002_bs_12_lamda_0.5.pth")
    parser.add_argument('--output_folder_name', type=str, default=None, help='output folder base name')
    parser.add_argument('--rad_range', type=list, default=[-15, 5], help='range of radiation values for scaling')
    parser.add_argument('--test_path', type=str, default=None)
    return parser


if __name__ == "__main__":

    parser = arg_parser()
    args = parser.parse_args()
    inverse_checkpoint_folder = os.path.join(args.data_path, 'checkpoints_inverse')
    output_folder_name = args.output_folder_name if args.output_folder_name is not None else 'NN_generated_antennas'
    output_folder = os.path.join(inverse_checkpoint_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    device = torch.device("cpu")
    print(args, device)
    scaler_name = 'scaler'
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
    ant_scaler_manager.try_loading_from_cache()

    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=24000, repr_mode='abs')
    for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
        ant_abs_trn = torch.tensor(EMBEDDINGS, device=device)
        break
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=280, algorithm='auto').fit(ant_abs_trn)

    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1, repr_mode='abs')
    antenna_dataset_loader.load_test_data(args.test_path) if args.test_path is not None else None
    path = args.test_path if args.test_path is not None else data_path
    loader = antenna_dataset_loader.tst_loader if args.test_path is not None else antenna_dataset_loader.val_loader
    print('number of examples in train: ', len(antenna_dataset_loader.trn_folders))
    ant_out_dim = antenna_dataset_loader.trn_dataset.shapes['ant'][0]
    model = inverse_forward_concat(forw_module=forward_GammaRad(radiation_channels=12, rad_range=args.rad_range),
                                   inv_module=GammaRadHyperEnv(shapes={"fc1.inp_dim": 512, "fc1.out_dim": ant_out_dim}),
                                   )
    for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
        model.to(device)
        EMBEDDINGS, GAMMA, RADIATION, ENV, _ = sample
        embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
            GAMMA.to(device), RADIATION.to(device), \
            env_scaler_manager.scaler.forward(ENV).float().to(device)
        gamma_pred, rad_pred, ant = model(gamma, radiation, env)
        break
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    with torch.no_grad():
        model.eval()
        for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(loader):
            print(f'evaluating antenna {name[0]}.')
            x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            env_og_rel_repr = env_to_dict_representation(
                torch.tensor(np.load(os.path.join(path, name[0], 'environment.npy'))[np.newaxis]))[0]
            gamma_pred, rad_pred, ant_pred = model(gamma, rad, env)
            gamma_stats = produce_gamma_stats(gamma, gamma_to_dB(gamma_pred), dataset_type='dB')
            rad_stats = produce_radiation_stats(rad, rad_pred)
            validity_function = AntValidityFunction(sample_path=os.path.join(path, name[0]), ant_scaler=ant_scaler_manager)
            distances, neighbor_indices = nbrs.kneighbors(ant_scaler_manager.scaler.inverse(ant_pred))
            nbrs_abs_trn = ant_abs_trn[neighbor_indices[0]]
            nbrs_scaled_trn = ant_scaler_manager.scaler.forward(nbrs_abs_trn)
            validity = [validity_function(ant.unsqueeze(0)) for ant in nbrs_scaled_trn]
            valid_nbrs = np.nonzero(validity)[0]
            if len(valid_nbrs) == 0:
                print(f"No valid for neighbors {name[0]}, skipping")
                continue
            print(f"Found {len(valid_nbrs)} neighbors for {name[0]}. in index {valid_nbrs[0]}")
            best_nbr_abs = nbrs_abs_trn[valid_nbrs][0:1]
            best_nbr_og_repr = ant_abs2rel(ant_to_dict_representation(best_nbr_abs)[0], env_og_rel_repr)
            with open(os.path.join(output_folder, f'ant_{name[0]}_shahar.pickle'), 'wb') as ant_handle:
                pickle.dump(best_nbr_og_repr, ant_handle)
            with open(os.path.join(output_folder, f'env_{name[0]}_shahar.pickle'), 'wb') as env_handle:
                pickle.dump(env_og_rel_repr, env_handle)


