import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from io import BytesIO
import argparse
from models.forward_GammaRad import forward_GammaRad
from utils import *
from AntennaDesign.utils import *
from forward_model_evaluate_main import plot_condition


def get_valid_indices(antennas: np.ndarray, environment: dict) -> np.ndarray:
    is_valid = np.array([], dtype=bool)
    for ant in antennas:
        is_valid = np.append(is_valid, check_ant_validity(ant, environment))
    valid_indices = np.nonzero(is_valid)[0]
    print(f'Found {len(valid_indices)} valid antennas out of {len(antennas)} samples.')
    return valid_indices


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


def get_stats_for_top_k(k: int, sorting_indices: torch.Tensor, stats: tuple):
    appended_stats = []
    while len(appended_stats) < k:
        i = len(appended_stats)
        stats_i = tuple([stat[sorting_indices[i]] for stat in stats])
        appended_stats.append(stats_i)
    return appended_stats


def model_init_shape(model, data_loader: AntennaDataSetsLoader, device=torch.device("cpu")):
    with torch.no_grad():
        for idx, (EMBEDDINGS, _, _, ENV, _) in enumerate(data_loader.trn_loader):
            embeddings, env = torch.ones_like(EMBEDDINGS).to(device), torch.ones_like(ENV).to(device)
            geometry = torch.cat((embeddings, env), dim=1)
            _, _ = model(geometry)
            break


def extend_to_fit_samples(num_samples: int, env: torch.Tensor, gamma: torch.Tensor, rad: torch.Tensor):
    gamma = torch.tile(gamma, (num_samples, 1))
    rad = torch.tile(rad, (num_samples, 1, 1, 1))
    env = torch.tile(env, (num_samples, 1))
    return gamma, rad, env


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--forward_checkpoint_path', type=str, help='path to forward checkpoint',
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints\updated_forward_best_dict.pth')
    parser.add_argument('--samples_folder_name', type=str, default=None, help='folder base name for samples')
    parser.add_argument('--output_folder_name', type=str, default=None, help='output folder base name')
    parser.add_argument('--repr_mode', type=str, help='use relative or absolute repr. for ant and env', default='abs')
    return parser


if __name__ == "__main__":
    args = arg_parser().parse_args()
    device = torch.device("cpu")
    data_path = args.data_path
    inverse_checkpoint_folder = os.path.join(args.data_path, 'checkpoints_inverse')
    output_folder_name = args.output_folder_name if args.output_folder_name is not None else 'generated_antennas'
    output_folder = os.path.join(inverse_checkpoint_folder, output_folder_name)
    os.makedirs(output_folder, exist_ok=True)
    samples_folder_name = args.samples_folder_name if args.samples_folder_name is not None else 'samples'
    samples_folder = os.path.join(inverse_checkpoint_folder, samples_folder_name)
    samples_names = os.listdir(samples_folder)

    antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=1)
    antenna_dataset_loader.load_test_data(args.test_path) if args.test_path is not None else None
    path = args.test_path if args.test_path is not None else data_path
    loader = antenna_dataset_loader.tst_loader if args.test_path is not None else antenna_dataset_loader.val_loader

    scaler_name = 'scaler' if args.repr_mode == 'abs' else 'scaler_rel'
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    model = forward_GammaRad(radiation_channels=12)
    model_init_shape(model, antenna_dataset_loader)
    model.load_state_dict(torch.load(args.forward_checkpoint_path, map_location=device))
    counted = np.zeros(2)
    with torch.no_grad():
        k = 1
        all_gamma_stats, all_radiation_stats = [], []
        plot_GT_vs_pred = True
        model.eval()
        for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(loader):
            if all([name[0] not in sample_name for sample_name in samples_names]):
                    #or GAMMA[:, :GAMMA.shape[1] // 2].min() > -5:
                print(f'skipping antenna {name[0]} from the loader')
                continue
            print(f'evaluating samples for antenna {name[0]}.')
            x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)

            samples = torch.tensor(np.load(os.path.join(samples_folder, f'sample_{name[0]}.npy'))).float().to(device)

            env_og_rel_repr = env_to_dict_representation(
                torch.tensor(np.load(os.path.join(path, name[0], 'environment.npy'))[np.newaxis]))[0]
            gt_ant_og_repr = ant_to_dict_representation(ant_scaler_manager.scaler.inverse(x))[0]
            samples_og_repr = ant_to_dict_representation(ant_scaler_manager.scaler.inverse(samples))
            assert args.repr_mode == 'abs',\
                'Only absolute representation is supported for now. rel mode is not supported.'
            if args.repr_mode == 'abs':
                #making sure the representation is relative
                gt_ant_og_repr = ant_abs2rel(gt_ant_og_repr, env_og_rel_repr)
                samples_og_repr = [ant_abs2rel(ant, env_og_rel_repr) for ant in samples_og_repr]

            valid_samples_indices = get_valid_indices(samples_og_repr, env_og_rel_repr)
            if len(valid_samples_indices) < 3:
                print(f'No valid samples for antenna {name[0]}.')
                continue
            valid_samples = samples[valid_samples_indices]

            num_samples = valid_samples.shape[0]
            env_tiled = torch.tile(env, (num_samples, 1))
            geometries = torch.cat((valid_samples, env_tiled), dim=1)
            gamma_pred, rad_pred = model(geometries)
            gamma_pred_dB = gamma_to_dB(gamma_pred)
            gamma_stats = produce_gamma_stats(gamma, gamma_pred_dB, dataset_type='dB', to_print=False)
            radiation_stats = produce_radiation_stats(rad, rad_pred, to_print=False)
            sorting_idxs = sort_by_metric(*gamma_stats, *radiation_stats)
            gamma_pred_dB_sorted = gamma_pred_dB[sorting_idxs]
            rad_pred_sorted = rad_pred[sorting_idxs]
            samples_sorted = valid_samples[sorting_idxs]
            top_k_gamma_stats = get_stats_for_top_k(k, sorting_idxs, gamma_stats)
            top_k_radiation_stats = get_stats_for_top_k(k, sorting_idxs, radiation_stats)
            print(f'For antenna {name[0]}: \n',f'gamma top {k} stats:', top_k_gamma_stats,
                  f'\nradiation top {k} stats', top_k_radiation_stats)
            all_gamma_stats.append(top_k_gamma_stats)
            all_radiation_stats.append(top_k_radiation_stats)
            samples_sorted_og_repr = ant_to_dict_representation(ant_scaler_manager.scaler.inverse(samples_sorted))
            if args.repr_mode == 'abs':
                samples_sorted_og_repr = [ant_abs2rel(ant, env_og_rel_repr) for ant in samples_sorted_og_repr]
            if plot_GT_vs_pred:
                figs_gt, axs_gt = plt.subplots(2, 1, figsize=(5, 10))
                fig_gt = plot_condition((gamma, rad),  plot_type='2d')
                img_gt = figure_to_image(fig_gt)
                plt.close(fig_gt)
                axs_gt[1].imshow(img_gt)
                axs_gt[1].set_title('GT target')
                axs_gt[1].axis('off')
                antenna_fig = plot_antenna_figure(ant_parameters=gt_ant_og_repr, model_parameters=env_og_rel_repr)
                antenna_im = figure_to_image(antenna_fig)
                plt.close(antenna_fig)
                axs_gt[0].imshow(antenna_im)
                axs_gt[0].set_title('GT Antenna')
                axs_gt[0].axis('off')

                fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                # save env_og_repr
                with open(os.path.join(output_folder, f'env_{name[0]}.pickle'), 'wb') as env_handle:
                    pickle.dump(env_og_rel_repr, env_handle)
                for i in [0, 1, 2]:
                    ant_best_og_repr = samples_sorted_og_repr[i]
                    counted[check_ant_validity(ant_best_og_repr, env_og_rel_repr)] += 1
                    with open(os.path.join(output_folder, f'ant_{name[0]}_grade_{i}.pickle'),
                              'wb') as ant_handle:
                        pickle.dump(ant_best_og_repr, ant_handle)

                    gamma_pred_dB_best = gamma_pred_dB_sorted[i].unsqueeze(0)
                    rad_pred_best = rad_pred_sorted[i].unsqueeze(0)

                    fig_pred = plot_condition((gamma_pred_dB_best, rad_pred_best), plot_type='2d')
                    img_pred = figure_to_image(fig_pred)
                    plt.close(fig_pred)
                    axs[1, i].imshow(img_pred)
                    axs[1, i].set_title(f'Generated Prediction, idx: {i}')
                    axs[1, i].axis('off')  # Turn off axes for better visualization
                    antenna_fig = plot_antenna_figure(ant_parameters=ant_best_og_repr, model_parameters=env_og_rel_repr)
                    antenna_im = figure_to_image(antenna_fig)
                    plt.close(antenna_fig)
                    axs[0, i].imshow(antenna_im)
                    axs[0, i].set_title(f'Generated Antenna, idx: {i}')
                    axs[0, i].axis('off')

                # plt.tight_layout()
                # plt.show()

        gamma_stats_final = np.array(all_gamma_stats)
        radiation_stats_final = np.array(all_radiation_stats)
        gamma_stats_final = gamma_stats_final.reshape(-1, gamma_stats_final.shape[-1])
        radiation_stats_final = radiation_stats_final.reshape(-1, radiation_stats_final.shape[-1])
        produce_stats_all_dataset(gamma_stats_final, radiation_stats_final)
