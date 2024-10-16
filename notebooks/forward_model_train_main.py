from models.forward_GammaRad import forward_GammaRad
from losses import GammaRad_loss
from AntennaDesign.utils import *

import argparse
import torch
import os
import pickle


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_110k_150k_processed')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--gamma_schedule', type=float, default=0.95, help='gamma decay rate')
    parser.add_argument('--step_size', type=int, default=1, help='step size for gamma decay')
    parser.add_argument('--rad_range', type=list, default=[-20, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=0., help='controls the influence of geometry loss')
    parser.add_argument('--rad_phase_fac', type=float, default=0., help='weight for phase in radiation loss')
    parser.add_argument('--lamda', type=float, default=1., help='weight for radiation in gamma radiation loss')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to save model checkpoints')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--try_cache', action='store_true', help='try to load from cache')
    return parser.parse_args()


def save_embeddings(pca, data_path):
    for idx in sorted(os.listdir(data_path)):
        if os.path.exists(os.path.join(data_path, idx.zfill(5), 'embeddings.npy')) or any(c.isalpha() for c in idx):
            continue
        antenna = np.load(os.path.join(data_path, idx.zfill(5), 'antenna.npy'))
        ant_resized = cv2.resize(antenna, (200, 144))
        embeddings = pca.transform(ant_resized.flatten().reshape(1, -1)).flatten()
        np.save(os.path.join(data_path, idx.zfill(5), 'embeddings.npy'), embeddings)
        print(f'Saved embeddings for antenna {idx}.')


if __name__ == "__main__":
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args, device)
    # pca = pickle.load(open(os.path.join(args.data_path, 'pca_model.pkl'), 'rb'))
    antenna_dataset_loader = AntennaDataSetsLoader(args.data_path, batch_size=args.batch_size, try_cache=args.try_cache)
    print('number of examples in train: ', len(antenna_dataset_loader.trn_folders))
    model = forward_GammaRad(radiation_channels=12)
    loss_fn = GammaRad_loss(geo_weight=args.geo_weight, lamda=args.lamda, rad_phase_fac=args.rad_phase_fac, euc_weight=0.5)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_schedule)
    keep_training = True
    epoch = 0
    patience = args.patience
    checkpoints_path = os.path.join(args.data_path, 'checkpoints') if args.checkpoint_path is None else args.checkpoint_path
    best_loss = np.inf
    train_loss = 0
    env_scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'env_scaler.pkl'))
    env_scaler_manager.try_loading_from_cache()
    ant_scaler_manager = ScalerManager(path=os.path.join(args.data_path, 'ant_scaler.pkl'))
    ant_scaler_manager.try_loading_from_cache()
    while keep_training:
        if epoch % 10 == 0 and epoch > 0:
            print(f'Saving model at epoch {epoch}')
            torch.save(model.state_dict(), os.path.join(checkpoint_path, f'forward_epoch{epoch}.pth'))

        print(f'Starting Epoch: {epoch}. Patience: {patience}')
        model.train()
        for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV, name = sample
            embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device),\
                GAMMA.to(device), RADIATION.to(device), \
                env_scaler_manager.scaler.forward(ENV).float().to(device)
            geometry = torch.cat((embeddings, env), dim=1)
            target = (gamma, radiation)
            optimizer.zero_grad()
            gamma_pred, rad_pred = model(geometry)
            output = (gamma_pred, rad_pred, geometry)
            loss = loss_fn(output, target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {idx}, Loss: {train_loss / (idx + 1)}')
        train_loss /= len(antenna_dataset_loader.trn_loader)
        print(f'End Epoch: {epoch}, Loss: {train_loss}')
        epoch += 1
        lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {lr}')
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for idx, sample in enumerate(antenna_dataset_loader.val_loader):
                EMBEDDINGS, GAMMA, RADIATION, ENV, name = sample
                embeddings, gamma, radiation, env = ant_scaler_manager.scaler.forward(EMBEDDINGS).float().to(device), \
                    GAMMA.to(device), RADIATION.to(device), \
                    env_scaler_manager.scaler.forward(ENV).float().to(device)
                geometry = torch.cat((embeddings, env), dim=1)
                target = (gamma, radiation)
                gamma_pred, rad_pred = model(geometry)
                output = (gamma_pred, rad_pred, geometry)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            val_loss /= len(antenna_dataset_loader.val_loader)
            print(f'Validation Loss: {val_loss}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
                patience = args.patience
            else:
                patience -= 1
            if patience <= np.ceil(args.patience / 2):
                scheduler.step()
            if patience == 0:
                print('Early stopping - stayed at the same loss for too long.')
                keep_training = False
            train_loss = 0
    best_model_checkpoint_path = os.path.join(checkpoints_path, 'forward_best_dict.pth')
    torch.save(best_model.state_dict(), best_model_checkpoint_path)
    print('Training finished.')
    print(f'Best loss: {best_loss}')
    print(f'Best model saved at: {best_model_checkpoint_path}')
