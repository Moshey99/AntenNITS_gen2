import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../AntennaDesign')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from AntennaDesign.models.forward_GammaRad import forward_GammaRad
from AntennaDesign.models.inverse_hypernet import inverse_forward_concat
from nits.antenna_condition import GammaRadHyperEnv
from losses import GammaRad_loss, Euclidean_GammaRad_Loss
from AntennaDesign.utils import *

import argparse
import torch
import pickle


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k')
    parser.add_argument('--forward_checkpoint_path', type=str,
                        default=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k\checkpoints\updated_forward_best_dict.pth")
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--gamma_schedule', type=float, default=0.95, help='gamma decay rate')
    parser.add_argument('--step_size', type=int, default=1, help='step size for gamma decay')
    parser.add_argument('--rad_range', type=list, default=[-15, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=0., help='controls the influence of geometry loss')
    parser.add_argument('--euc_weight', type=float, default=0., help='weight for euclidean loss in GammaRad loss')
    parser.add_argument('--rad_phase_fac', type=float, default=0., help='weight for phase in radiation loss')
    parser.add_argument('--lamda', type=float, default=0.5, help='weight for radiation in gamma radiation loss')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to save model checkpoints')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--repr_mode', type=str, help='use relative repr. for ant and env', default='abs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(args, device)
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=args.batch_size, repr_mode=args.repr_mode)
    print('number of examples in train: ', len(antenna_dataset_loader.trn_folders))
    ant_out_dim = antenna_dataset_loader.trn_dataset.shapes['ant'][0]
    model = inverse_forward_concat(forw_module=forward_GammaRad(radiation_channels=12, rad_range=args.rad_range),
                                   inv_module=GammaRadHyperEnv(shapes={"fc1.inp_dim": 512, "fc1.out_dim": ant_out_dim}),
                                   )
    loss_fn = GammaRad_loss(geo_weight=args.geo_weight, lamda=args.lamda,
                            rad_phase_fac=args.rad_phase_fac, euc_weight=args.euc_weight)
    scaler_name = 'scaler' if args.repr_mode == 'abs' else 'scaler_rel'
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
    model.load_and_freeze_forward(args.forward_checkpoint_path)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_schedule)
    keep_training = True
    epoch = 0
    patience = args.patience
    checkpoints_path = os.path.join(args.data_path, 'checkpoints') if args.checkpoint_path is None else args.checkpoint_path
    os.makedirs(checkpoints_path, exist_ok=True)
    best_loss = np.inf
    train_loss = 0
    scaler_name = 'scaler' if args.repr_mode == 'abs' else 'scaler_rel'
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'env_{scaler_name}.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, f'ant_{scaler_name}.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    while keep_training:
        if epoch % 10 == 0 and epoch > 0:
            print(f'Saving model at epoch {epoch}')
            torch.save(model.state_dict(), os.path.join(checkpoints_path, f'inv_forward_epoch_{epoch}_lr_{args.lr}_bs_{args.batch_size}.pth'))

        print(f'Starting Epoch: {epoch}. Patience: {patience}')
        model.train()
        for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
            ANT, GAMMA, RADIATION, ENV, name = sample
            ant, gamma, radiation, env = ant_scaler_manager.scaler.forward(ANT).float().to(device),\
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            target = (gamma, radiation)
            optimizer.zero_grad()
            gamma_pred, rad_pred, ant_pred = model(gamma, radiation, env)
            geometry = torch.cat((ant_pred, env), dim=1)
            output = (gamma_pred, rad_pred, geometry)
            loss = loss_fn(output, target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {idx}, Loss: {train_loss / (idx + 1)}')
                produce_gamma_stats(gamma, gamma_to_dB(gamma_pred), dataset_type='dB', to_print=True)
                produce_radiation_stats(radiation, rad_pred, to_print=True)
        train_loss /= len(antenna_dataset_loader.trn_loader)
        print(f'End Epoch: {epoch}, Loss: {train_loss}')
        epoch += 1
        lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {lr}')
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for idx, sample in enumerate(antenna_dataset_loader.val_loader):
                ANT, GAMMA, RADIATION, ENV, name = sample
                ant, gamma, radiation, env = ant_scaler_manager.scaler.forward(ANT).float().to(device), \
                    GAMMA.to(device), RADIATION.to(device), \
                    env_scaler_manager.scaler.forward(ENV).float().to(device)
                target = (gamma, radiation)
                gamma_pred, rad_pred, ant_pred = model(gamma, radiation, env)
                geometry = torch.cat((ant_pred, env), dim=1)
                produce_gamma_stats(gamma, gamma_to_dB(gamma_pred), dataset_type='dB', to_print=True)
                produce_radiation_stats(radiation, rad_pred, to_print=True)
                output = (gamma_pred, rad_pred, geometry)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            val_loss /= len(antenna_dataset_loader.val_loader)
            print(f'Validation Loss: {val_loss}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = copy.deepcopy(model)
                patience = args.patience
            else:
                patience -= 1
            if patience <= np.ceil(args.patience / 2):
                scheduler.step()
            if patience == 0:
                print('Early stopping - stayed at the same loss for too long.')
                keep_training = False
            train_loss = 0
    best_model = best_model if best_model is not None else model
    best_model_checkpoint_path = os.path.join(checkpoints_path, f'inv_forward_best_dict_bestloss_{best_loss}_lr_{args.lr}_bs_{args.batch_size}_lamda_{args.lamda}.pth')
    torch.save(best_model.state_dict(), best_model_checkpoint_path)
    print('Training finished.')
    print(f'Best loss: {best_loss}')
    print(f'Best model saved at: {best_model_checkpoint_path}')
