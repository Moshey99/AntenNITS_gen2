import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import copy
import itertools
import sys
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
from notebooks.forward_model_evaluate_main import plot_condition


def list_str_to_list(s):
    print(s)
    assert s[0] == '[' and s[-1] == ']'
    s = s[1:-1]
    s = s.replace(' ', '')
    s = s.split(',')

    s = [int(x) for x in s]

    return s


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
parser.add_argument('--test_path', type=str, default=None)
parser.add_argument('--checkpoint_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints_inverse\ANT_model_lr_0.0002_hd_512_nr_8_pd_0.95_bs_12_drp_0.3_bounds_-3_3_INFO_updated_directivity.pth')
parser.add_argument('-o', '--output_folder', type=str, default=None)
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-hi', '--hidden_dim', type=int, default=512)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=8)
parser.add_argument('-n', '--patience', type=int, default=10)
parser.add_argument('-ga', '--gamma', type=float, default=0.9)
parser.add_argument('-pd', '--polyak_decay', type=float, default=0.9)
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
parser.add_argument('-l', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-p', '--dropout', type=float, default=-1.0)
parser.add_argument('-rc', '--add_residual_connections', type=bool, default=True)
parser.add_argument('-bm', '--bound_multiplier', type=int, default=1)
parser.add_argument('-w', '--step_weights', type=list_str_to_list, default='[1]',
                    help='Weights for each step of multistep NITS')
parser.add_argument('--bounds', type=list_str_to_list, default='[-3,3]')
parser.add_argument('--conditional', type=bool, default=True)
parser.add_argument('--conditional_dim', type=int, default=512)
parser.add_argument('--repr_mode', type=str, help='use relative repr. for ant and env', default='abs')
parser.add_argument('--num_samples', type=int, default=10)
parser.add_argument('--num_skip', type=int, default=0)
args = parser.parse_args()
output_folder = os.path.join(args.data_path,
                             'checkpoints_inverse', 'samples') if args.output_folder is None else args.output_folder
Path(output_folder).mkdir(parents=True, exist_ok=True)
conditional = args.conditional
default_dropout = 0
args.dropout = args.dropout if args.dropout >= 0.0 else default_dropout
use_batch_norm = True
print(args)

max_vals_ll = []
lasts_train_ll = []

step_weights = np.array(args.step_weights)
step_weights = step_weights / (np.sum(step_weights) + 1e-7)

if args.gpu != '':
    devices = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu.split(',')]
else:
    devices = ['cpu']
device = devices[0]

data_path = args.data_path
assert os.path.exists(data_path), f'data_path in {data_path} does not exist'
antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=1)
antenna_dataset_loader.load_test_data(args.test_path) if args.test_path is not None else None
loader = antenna_dataset_loader.tst_loader if args.test_path is not None else antenna_dataset_loader.val_loader
shapes = antenna_dataset_loader.trn_dataset.shapes
scaler_name = 'scaler' if args.repr_mode == 'abs' else 'scaler_rel'
env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
env_scaler_manager.try_loading_from_cache()
ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
ant_scaler_manager.try_loading_from_cache()

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
    nits_model=nits_model,
    n_residual_blocks=args.n_residual_blocks,
    hidden_dim=args.hidden_dim,
    dropout_probability=args.dropout,
    use_batch_norm=use_batch_norm,
    nits_input_dim=nits_input_dim,
    conditional=conditional,
    conditional_dim=args.conditional_dim)

shadow = Model(
    d=d,
    nits_model=nits_model,
    n_residual_blocks=args.n_residual_blocks,
    hidden_dim=args.hidden_dim,
    dropout_probability=args.dropout,
    use_batch_norm=use_batch_norm,
    nits_input_dim=nits_input_dim,
    conditional=conditional,
    conditional_dim=args.conditional_dim)

model = EMA(model, shadow, decay=args.polyak_decay).to(device)
model.eval()
with torch.no_grad():
    for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
        x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
            GAMMA.to(device), RADIATION.to(device), \
            env_scaler_manager.scaler.forward(ENV).float().to(device)
        condition = (gamma, rad, env)
        model.init_models_architecture(x, condition)
        break
checkpoint_path = args.checkpoint_path
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
num_samples = args.num_samples
with torch.no_grad():
    for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(loader):
        if idx < args.num_skip:
            print('skipping antenna: ', name[0])
            continue
        x, gamma, rad, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
            GAMMA.to(device), RADIATION.to(device), \
            env_scaler_manager.scaler.forward(ENV).float().to(device)
        print('Working on antenna: ', name[0])
        # plot_condition((gamma, rad))
        # plt.show()
        # condition = (gamma, rad, env)
        # print(f'sampling {args.num_samples} samples for antenna: ', name[0])
        # start = time.time()
        # smp = model.shadow.sample(num_samples, device, condition=condition)
        # print(f'sampled {num_samples} samples in {time.time() - start} seconds.')
        # np.save(os.path.join(output_folder, f'sample_{name[0]}.npy'), smp.detach().cpu().numpy())
