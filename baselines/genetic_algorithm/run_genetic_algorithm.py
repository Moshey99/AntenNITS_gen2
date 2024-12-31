import os
import sys
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from baselines.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from AntennaDesign.losses import GammaRad_loss
from AntennaDesign.utils import *
from notebooks.forward_model_evaluate_main import plot_condition
from models.forward_GammaRad import forward_GammaRad


import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple, Callable


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--rad_range', type=list, default=[-15, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=0., help='controls the influence of geometry loss')
    parser.add_argument('--euc_weight', type=float, default=0., help='weight for euclidean loss in GammaRad loss')
    parser.add_argument('--rad_phase_fac', type=float, default=0., help='weight for phase in radiation loss')
    parser.add_argument('--lamda', type=float, default=0.5, help='weight for radiation in gamma radiation loss')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints\updated_forward_best_dict.pth")
    parser.add_argument('--repr_mode', type=str, help='use relative repr. for ant and env', default='abs')
    parser.add_argument('--output_folder_name', type=str, default=None, help='output folder base name')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    return parser.parse_args()


class FitnessFunction:
    def __init__(
        self,
        model: Callable[[torch.Tensor], torch.Tensor],
        loss_func: Callable[[tuple, tuple], torch.Tensor],
        target: Tuple,
        env: torch.Tensor
    ) -> None:
        self.model = model
        self.loss_func = loss_func
        self.target = target
        self.env = env

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        all_fitnesses = []
        env_tiled = torch.tile(self.env, dims=(x.shape[0], 1))
        geo = torch.cat((x, env_tiled), dim=1)
        pred = self.model(geo)
        pred_with_geo_tuple = pred + (geo,)
        for i in range(len(x)):
            specific_pred = tuple(v[i].unsqueeze(0) for v in pred_with_geo_tuple)
            fitness = -self.loss_func(specific_pred, self.target)
            all_fitnesses.append(fitness)
        return torch.tensor(all_fitnesses)


class AntValidityFunction:
    def __init__(
        self,
        sample_path: str,
        ant_scaler: ScalerManager
    ) -> None:
        args.path = sample_path
        args.ant_scaler = ant_scaler

    def __call__(self, ant: torch.Tensor) -> bool:
        env_og_rel_repr = env_to_dict_representation(
            torch.tensor(np.load(os.path.join(args.path, 'environment.npy'))[np.newaxis]))[0]
        ant_abs = args.ant_scaler.scaler.inverse(ant)
        ant_og_abs_repr = ant_to_dict_representation(ant_abs)[0]
        ant_og_rel_repr = ant_abs2rel(ant_og_abs_repr, env_og_rel_repr)
        return bool(check_ant_validity(ant_og_rel_repr, env_og_rel_repr))


if __name__ == "__main__":
    all_gamma_stats, all_radiation_stats = [], []
    plot_GT_vs_pred = False
    args = arg_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(args, device)
    inverse_checkpoint_folder = os.path.join(args.data_path, 'checkpoints_inverse')
    output_folder_name = args.output_folder_name if args.output_folder_name is not None else 'generated_antennas'
    output_folder = os.path.join(inverse_checkpoint_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=1, repr_mode=args.repr_mode)
    antenna_dataset_loader.load_test_data(args.test_path) if args.test_path is not None else None
    loader = antenna_dataset_loader.tst_loader if args.test_path is not None else antenna_dataset_loader.val_loader
    model = forward_GammaRad(radiation_channels=12, rad_range=args.rad_range)
    loss_fn = GammaRad_loss(geo_weight=args.geo_weight, lamda=args.lamda,
                            rad_phase_fac=args.rad_phase_fac, euc_weight=args.euc_weight)
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
        target = (gamma, radiation)

        geometry = torch.cat((embeddings, env), dim=1)
        gamma_pred, rad_pred = model(geometry)
        break
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    with torch.no_grad():
        model.eval()
        for idx, sample in enumerate(loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV, name = sample
            embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            print(f'Working on antenna #{name[0]}')
            sample_path = args.test_path if args.test_path is not None else args.data_path
            validity_function = AntValidityFunction(os.path.join(sample_path, name[0]), ant_scaler_manager)
            if not validity_function(embeddings):
                warnings.warn('Validation function is not correct. Embeddings expected to be valid for its own environment')
            target = (gamma, radiation)
            nearest_neighbors_path = os.path.join(args.test_path, name[0], 'valid_neighbors_scaled.pth')
            nearest_neighbors = torch.load(nearest_neighbors_path)
            fitness_func = FitnessFunction(model, loss_fn, target, env)
            population_size = min(150, nearest_neighbors.shape[0])
            ga = GeneticAlgorithm(
                vector_length=40,
                initial_population=nearest_neighbors[:population_size],
                population_size=population_size,
                generations=30,
                mutation_stddev=0.02,
                fitness_function=fitness_func,
            )
            best_ant, best_loss = ga.run(validity_function)
            with open(os.path.join(output_folder, f'ant_{name[0]}.pickle'),
                      'wb') as ant_handle:
                env_og_rel_repr = env_to_dict_representation(
                    torch.tensor(np.load(os.path.join(args.path, 'environment.npy'))[np.newaxis]))[0]
                with open(os.path.join(output_folder, f'env_{name[0]}.pickle'), 'wb') as env_handle:
                    pickle.dump(env_og_rel_repr, env_handle)
                best_ant_abs = args.ant_scaler.scaler.inverse(best_ant)
                best_ant_og_abs_repr = ant_to_dict_representation(best_ant_abs)[0]
                best_ant_og_rel_repr = ant_abs2rel(best_ant_og_abs_repr, env_og_rel_repr)
                pickle.dump(best_ant_og_rel_repr, ant_handle)
            geometry_pred = torch.cat((best_ant, env), dim=1)
            gamma_pred, rad_pred = model(geometry_pred)
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