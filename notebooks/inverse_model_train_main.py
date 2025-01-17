import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import copy
import itertools
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from nits.model import *
from nits.layer import *
from nits.fc_model import *
from nits.cnn_model import *
from nits.resmade import ResidualMADE
from nits.fc_model import ResMADEModel
from scipy.stats import gaussian_kde
from scipy.special import kl_div
from pathlib import Path
import glob
from AntennaDesign.utils import *
from AntennaDesign.models.forward_GammaRad import forward_GammaRad

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def correlation_dist(corr_mat1, corr_mat2):
    d = 1 - np.trace(np.dot(corr_mat1, corr_mat2)) / (np.linalg.norm(corr_mat1) * np.linalg.norm(corr_mat2))
    return d


def kl_eval(model, data, n):
    paths = sorted(glob.glob('nits_checkpoints/ANTmodel_*.pth'))
    vv = np.linspace(-4, 4, num=7000)
    kl_divs = []
    for path in paths:
        kl_divs_path = []
        print(path)
        model.load_state_dict(torch.load(path, map_location=device))
        smp = model.model.sample(n, device)
        for feature in range(smp.shape[1]):
            kde_smp = gaussian_kde(smp[:, feature]).pdf(vv)
            kde_real = gaussian_kde(data.trn.x[:, feature]).pdf(vv)
            kl_div_value = np.sum(kl_div(kde_real, kde_smp))
            kl_divs_path.append(kl_div_value)
        kl_divs.append(kl_divs_path)
    kl_divs = np.array(kl_divs)
    kl_divs = np.sum(kl_divs, axis=1)
    best_path = paths[np.argmin(kl_divs)]
    print(f'best model for KL Divergence evaluation is {best_path}')
    return


def plot_condition(condition, freqs):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    gamma, rad = condition
    gamma_amp, gamma_phase = gamma[:, :gamma.shape[1] // 2], gamma[:, gamma.shape[1] // 2:]
    ax1.plot(freqs, gamma_amp[0].cpu().detach().numpy(), 'b-')
    ax11 = ax1.twinx()
    ax11.plot(freqs, gamma_phase[0].cpu().detach().numpy(), 'r-')
    ax2.imshow(rad[0, 0].cpu().detach().numpy())
    ax1.set_title('condition gamma')
    ax1.set_ylabel('amplitude', color='b')
    ax1.set_ylim([-20, 0])
    ax11.set_ylim([-np.pi, np.pi])
    ax11.set_ylabel('phase', color='r')
    ax2.set_title('condition radiation')
    return fig


def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s


def produce_NN_stats(data):
    def find_NN(X_train, query, k=5):
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X_train)
        distances, indices = nbrs.kneighbors(query)
        return indices

    k_neighbors = 5
    n_train_samples, n_val_samples = len(data.trn.x), len(data.val.x)
    antenna_nn_indices = find_NN(data.trn.x, data.val.x, k=k_neighbors)
    nn_gammas = data.trn.gamma[antenna_nn_indices].reshape(n_val_samples * k_neighbors, -1)
    nn_gammas = downsample_gamma(nn_gammas)
    nn_radiations = data.trn.radiation[antenna_nn_indices].reshape(n_val_samples * k_neighbors,
                                                                   *data.trn.radiation.shape[1:])
    nn_radiations = downsample_radiation(nn_radiations)
    gt_gammas = torch.tensor(np.repeat(downsample_gamma(data.val.gamma), k_neighbors, axis=0))
    gt_radiations = torch.tensor(np.repeat(downsample_radiation(data.val.radiation), k_neighbors, axis=0))
    produce_gamma_stats(GT_gamma=gt_gammas, predicted_gamma=torch.tensor(nn_gammas), dataset_type='dB')
    produce_radiation_stats(predicted_radiation=torch.tensor(nn_radiations), gt_radiation=gt_radiations)
    pass


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


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str,
                default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('-o', '--output_folder', type=str, default=None)
    parser.add_argument('-g', '--gpu', type=str, default='', help='comma-separated list of GPU IDs')
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-hi', '--hidden_dim', type=int, default=512)
    parser.add_argument('-nr', '--n_residual_blocks', type=int, default=8)
    parser.add_argument('-n', '--patience', type=int, default=10, help='epoch patience for early stopping')
    parser.add_argument('-ga', '--gamma', type=float, default=0.9, help='gamma parameter for scheduler step')
    parser.add_argument('-pd', '--polyak_decay', type=float, default=0.9, help='parameter for polyak average smoothing')
    parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]', help='architecture for the nits')
    parser.add_argument('-l', '--learning_rate', type=float, default=2e-4)
    parser.add_argument('-p', '--dropout', type=float, default=-1.0, help='dropout probability')
    parser.add_argument('--bounds', type=list_str_to_list, default='[-3,3]', help='bounds for the values of the antenna')
    parser.add_argument('--no-conditional', action='store_false', dest='conditional', help='Set to disable conditional mode')
    parser.add_argument('--conditional_dim', type=int, default=512, help='dimensionality of the condition')
    parser.add_argument('-cm', '--condition_mode', type=str, default="separated", help='mode of the condition backbone in NITS')
    parser.add_argument('--repr_mode', type=str, help='use relative repr. for ant and env', default='abs')
    parser.add_argument('--run_info', type=str, default='', help='run information')
    # these arguments are less used
    parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
    parser.add_argument('-w', '--step_weights', type=list_str_to_list, default='[1]',
                        help='Weights for each step of multistep NITS')
    parser.add_argument('-rc', '--add_residual_connections', type=bool, default=True)
    parser.add_argument('--scarf', action="store_true")
    parser.add_argument('-r', '--rotate', action='store_true')
    parser.add_argument('-bm', '--bound_multiplier', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    output_folder = os.path.join(args.data_path,
                                 'checkpoints_inverse') if args.output_folder is None else args.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    conditional = args.conditional
    lr_grid = [args.learning_rate]
    hidden_dim_grid = [args.hidden_dim]
    nr_blocks_grid = [args.n_residual_blocks]
    polyak_decay_grid = [args.polyak_decay]
    batch_size_grid = [args.batch_size]

    max_vals_ll = []
    lasts_train_ll = []
    model_names = []
    for lr, hidden_dim, nr_blocks, polyak_decay, bs in itertools.product(lr_grid, hidden_dim_grid, nr_blocks_grid,
                                                                         polyak_decay_grid, batch_size_grid):
        model_extra_string = f'lr_{lr}_hd_{hidden_dim}_nr_{nr_blocks}_pd_{polyak_decay}_bs_{bs}_info_{args.run_info}_cm_{args.condition_mode}'
        model_names.append(model_extra_string)
        print(model_extra_string)
        args.learning_rate = lr
        args.hidden_dim = hidden_dim
        args.n_residual_blocks = nr_blocks
        args.polyak_decay = polyak_decay
        args.batch_size = bs
        step_weights = np.array(args.step_weights)
        step_weights = step_weights / (np.sum(step_weights) + 1e-7)

        if args.gpu != '':
            devices = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu.split(',')]
        else:
            devices = ['cpu']
        device = devices[0]

        use_batch_norm = True
        zero_initialization = False
        default_patience = 10
        data_path = args.data_path
        assert os.path.exists(data_path)
        antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=args.batch_size)
        shapes = antenna_dataset_loader.trn_dataset.shapes
        print('number of examples in train: ', len(antenna_dataset_loader.trn_folders))

        default_dropout = 0
        args.patience = args.patience if args.patience >= 0 else default_patience
        args.dropout = args.dropout if args.dropout >= 0.0 else default_dropout
        print(args)

        d = shapes['ant'][0]

        max_val = args.bounds[1]  # max(data.trn.x.max(), data.val.x.max(), data.tst.x.max())
        min_val = args.bounds[0]  # min(data.trn.x.min(), data.val.x.min(), data.tst.x.min())
        max_val, min_val = torch.tensor(max_val).to(device).float(), torch.tensor(min_val).to(device).float()
        max_val *= args.bound_multiplier
        min_val *= args.bound_multiplier

        nits_input_dim = [1]
        nits_model = NITS(d=d, arch=nits_input_dim + args.nits_arch, start=min_val, end=max_val, monotonic_const=1e-5,
                          A_constraint='neg_exp',
                          final_layer_constraint='softmax',
                          add_residual_connections=args.add_residual_connections,
                          normalize_inverse=(not args.dont_normalize_inverse),
                          softmax_temperature=False).to(device)

        model = Model(
            d=d,
            rotate=args.rotate,
            nits_model=nits_model,
            n_residual_blocks=args.n_residual_blocks,
            hidden_dim=args.hidden_dim,
            dropout_probability=args.dropout,
            use_batch_norm=use_batch_norm,
            zero_initialization=zero_initialization,
            nits_input_dim=nits_input_dim,
            conditional=conditional,
            conditional_dim=args.conditional_dim,
            condition_mode=args.condition_mode,
        )

        shadow = Model(
            d=d,
            rotate=args.rotate,
            nits_model=nits_model,
            n_residual_blocks=args.n_residual_blocks,
            hidden_dim=args.hidden_dim,
            dropout_probability=args.dropout,
            use_batch_norm=use_batch_norm,
            zero_initialization=zero_initialization,
            nits_input_dim=nits_input_dim,
            conditional=conditional,
            conditional_dim=args.conditional_dim,
            condition_mode=args.condition_mode
        )

        model = EMA(model, shadow, decay=args.polyak_decay).to(device)
        if len(devices) > 1:
            model = nn.DataParallel(model, device_ids=[int(i) for i in args.gpu.split(',')])

        # print number of parameters
        print('number of model parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
        print_every = 1
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=args.gamma)

        scaler_name = 'scaler' if args.repr_mode == 'abs' else 'scaler_rel'
        env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
        env_scaler_manager.try_loading_from_cache()
        ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
        ant_scaler_manager.try_loading_from_cache()

        time_ = time.time()
        epoch = 0
        train_ll = 0.
        max_val_ll = -np.inf
        patience = args.patience
        keep_training = True
        start_time = time.time()
        while keep_training:
            print('epoch', epoch, 'time [min]', round((time.time() - start_time) / 60), 'lr',
                  optim.param_groups[0]['lr'])
            model.train()
            for i, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
                x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                    GAMMA.to(device), RADIATION.to(device), \
                    env_scaler_manager.scaler.forward(ENV).float().to(device)
                model.init_models_architecture(x, (gamma, rad, env)) if i == 0 else None  # initialize the model
                # architecture, as its size is determined by the input
                ll = model(x, (gamma, rad, env))
                optim.zero_grad()
                (-ll).backward()
                train_ll += ll.detach().cpu().numpy()
                optim.step()
                model.update()
            epoch += 1
            print('current ll loss:', ll / len(x))
            if epoch % print_every == 0:
                # compute train loss
                train_ll /= len(antenna_dataset_loader.trn_dataset) * print_every
                lr = optim.param_groups[0]['lr']

                with torch.no_grad():
                    model.eval()
                    val_ll = 0.
                    ema_val_ll = 0.
                    for i, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.val_loader):
                        x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                            GAMMA.to(device), RADIATION.to(device), \
                            env_scaler_manager.scaler.forward(ENV).float().to(device)
                        val_ll += model.model(x, (gamma, rad, env)).detach().cpu().numpy()
                        ema_val_ll += model(x, (gamma, rad, env)).detach().cpu().numpy()

                    val_ll /= len(antenna_dataset_loader.val_dataset)
                    ema_val_ll /= len(antenna_dataset_loader.val_dataset)

                # early stopping
                if ema_val_ll > max_val_ll + 1e-4:
                    patience = args.patience
                    max_val_ll = ema_val_ll
                    best_model = copy.deepcopy(model)
                else:
                    patience -= 1
                print('Patience = ', patience)
                if patience <= args.patience - 2:
                    scheduler.step()
                if patience == 0:
                    print("Patience reached zero. max_val_ll stayed at {:.3f} for {:d} iterations.".format(max_val_ll,
                                                                                                           args.patience))
                    max_vals_ll.append(max_val_ll)
                    lasts_train_ll.append(train_ll)
                    keep_training = False

                test_ll = 0
                ema_test_ll = 0

                fmt_str1 = 'epoch: {:3d}, time: {:3d}s, train_ll: {:.4f},'
                fmt_str2 = ' ema_val_ll: {:.4f}, ema_test_ll: {:.4f},'
                fmt_str3 = ' val_ll: {:.4f}, test_ll: {:.4f}, lr: {:.2e}'

                print((fmt_str1 + fmt_str2 + fmt_str3).format(
                    epoch,
                    int(time.time() - time_),
                    train_ll,
                    ema_val_ll,
                    ema_test_ll,
                    val_ll,
                    test_ll,
                    lr))

                time_ = time.time()
                train_ll = 0.

            if epoch % (print_every * 10) == 0:
                # save best model so far
                torch.save(best_model.state_dict(), os.path.join(output_folder, f'ANT_model_{model_extra_string}.pth'))
                with open(os.path.join(output_folder, 'epoch.txt'), 'w') as f:
                    f.write(str(epoch))
                print(args)
    dict_to_print = {model_extra_string: (np.round(max_val, 3), np.round(last_train, 3)) for
                     model_extra_string, max_val, last_train in zip(model_names, max_vals_ll, lasts_train_ll)}
    min_idx_val = np.argmin(max_vals_ll)
    min_idx_train = np.argmin(lasts_train_ll)
    print(f'best model according to val set: {model_names[min_idx_val]}')
    print(f'best model according to train set: {model_names[min_idx_train]}')
    print('dict_to_print:', dict_to_print)
