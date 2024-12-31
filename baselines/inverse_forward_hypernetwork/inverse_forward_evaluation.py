import copy
import os
import sys
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
                        default=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints_inverse\inv_forward_best_dict.pth")
    parser.add_argument('--rad_range', type=list, default=[-15, 5], help='range of radiation values for scaling')
    parser.add_argument('--test_path', type=str, default=None)
    return parser


if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    device = torch.device("cpu")
    print(args, device)
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1, repr_mode='abs')
    antenna_dataset_loader.load_test_data(args.test_path) if args.test_path is not None else None
    path = args.test_path if args.test_path is not None else data_path
    loader = antenna_dataset_loader.tst_loader if args.test_path is not None else antenna_dataset_loader.val_loader
    print('number of examples in train: ', len(antenna_dataset_loader.trn_folders))
    ant_out_dim = antenna_dataset_loader.trn_dataset.shapes['ant'][0]
    model = inverse_forward_concat(forw_module=forward_GammaRad(radiation_channels=12, rad_range=args.rad_range),
                                   inv_module=GammaRadHyperEnv(shapes={"fc1.inp_dim": 512, "fc1.out_dim": ant_out_dim}),
                                   )
    scaler_name = 'scaler'
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    best_model = None
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
            gamma_pred, rad_pred, ant_pred = model(gamma, rad, env)
            gamma_stats = produce_gamma_stats(gamma, gamma_to_dB(gamma_pred), dataset_type='dB')
            rad_stats = produce_radiation_stats(rad, rad_pred)
            validity_function = AntValidityFunction(sample_path=os.path.join(path, name[0]), ant_scaler=ant_scaler_manager)
            is_valid = validity_function(ant_pred)
            print('valid? - {0}'.format(is_valid))


