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
import os
from pathlib import Path
import glob
from AntennaDesign.utils import *
from AntennaDesign.models.forward_GammaRad import forward_GammaRad


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
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
parser.add_argument('--checkpoint_path', type=str,
                    default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints_inverse\ANT_model_lr_0.0002_hd_256_nr_8_pd_0.975_bs_30.pth')
parser.add_argument('-o', '--output_folder', type=str, default=None)
parser.add_argument('-b', '--batch_size', type=int, default=30)
parser.add_argument('-g', '--gpu', type=str, default='')
parser.add_argument('-hi', '--hidden_dim', type=int, default=256)
parser.add_argument('-nr', '--n_residual_blocks', type=int, default=8)
parser.add_argument('-n', '--patience', type=int, default=10)
parser.add_argument('-ga', '--gamma', type=float, default=0.9)
parser.add_argument('-pd', '--polyak_decay', type=float, default=0.9995)
parser.add_argument('-a', '--nits_arch', type=list_str_to_list, default='[16,16,1]')
parser.add_argument('-dn', '--dont_normalize_inverse', type=bool, default=False)
parser.add_argument('-l', '--learning_rate', type=float, default=2e-4)
parser.add_argument('-p', '--dropout', type=float, default=-1.0)
parser.add_argument('-rc', '--add_residual_connections', type=bool, default=True)
parser.add_argument('-bm', '--bound_multiplier', type=int, default=1)
parser.add_argument('-w', '--step_weights', type=list_str_to_list, default='[1]',
                    help='Weights for each step of multistep NITS')
parser.add_argument('--scarf', action="store_true")
parser.add_argument('--bounds', type=list_str_to_list, default='[-10,10]')
parser.add_argument('--conditional', type=bool, default=True)
parser.add_argument('--conditional_dim', type=int, default=512)
parser.add_argument('--num_samples', type=int, default=1)
args = parser.parse_args()
start = time.time()
output_folder = os.path.join(args.data_path,
                             'checkpoints_inverse') if args.output_folder is None else args.output_folder
Path(output_folder).mkdir(parents=True, exist_ok=True)
conditional = args.conditional
default_dropout = 0
args.dropout = args.dropout if args.dropout >= 0.0 else default_dropout
use_batch_norm = True
print(args)

max_vals_ll = []
lasts_train_ll = []
model_extra_string = f'lr_{args.learning_rate}_hd_{args.hidden_dim}_nr_{args.n_residual_blocks}_pd_{args.polyak_decay}_bs_{args.batch_size}'
print(model_extra_string)
step_weights = np.array(args.step_weights)
step_weights = step_weights / (np.sum(step_weights) + 1e-7)

if args.gpu != '':
    devices = [torch.device('cuda:{}'.format(gpu)) for gpu in args.gpu.split(',')]
else:
    devices = ['cpu']
device = devices[0]

data_path = args.data_path
assert os.path.exists(data_path)
antenna_dataset_loader = AntennaDataSetsLoader(data_path, batch_size=1)
scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
scaler_manager.try_loading_from_cache()
if scaler_manager.scaler is None:
    raise ValueError('Scaler not found.')
print('number of examples in train: ', len(antenna_dataset_loader.trn_folders))

d = antenna_dataset_loader.embeddings_shape

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
        x, gamma, rad, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
            scaler_manager.scaler.forward(ENV).to(device)
        condition = (gamma, rad, env)
        model.init_models_architecture(x, condition)
        break
checkpoint_path = args.checkpoint_path
print('initialized model in: ', time.time() - start, ' seconds from the start')
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
print('loaded model in: ', time.time() - start, ' seconds from the start')
model.eval()
num_samples = args.num_samples
with torch.no_grad():
    for idx, (EMBEDDINGS, GAMMA, RADIATION, ENV, name) in enumerate(antenna_dataset_loader.trn_loader):
        x, gamma, rad, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
            scaler_manager.scaler.forward(ENV).to(device)
        condition = (gamma, rad, env)
        # print('preparing to sample, passed time: ', time.time() - start, ' seconds from the start')
        # smp = model.shadow.sample(num_samples, device, condition=condition)
        # print(f'sampled {num_samples} samples. passed time: ', time.time() - start, ' seconds from the start')
        # np.save(os.path.join(output_folder, f'sample_{name[0]}.npy'), smp.detach().cpu().numpy())
