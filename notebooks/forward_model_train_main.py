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
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs')
    parser.add_argument('--forward_model_path_gamma', type=str,
                        default=r'checkpoints/forward_gamma_smoothness_0.001_0.0001.pth')
    parser.add_argument('--forward_model_path_radiation', type=str,
                        default=r'checkpoints/forward_radiation_huberloss.pth')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--gamma_schedule', type=float, default=0.95, help='gamma decay rate')
    parser.add_argument('--step_size', type=int, default=1, help='step size for gamma decay')
    parser.add_argument('--rad_range', type=list, default=[-55, 5], help='range of radiation values for scaling')
    parser.add_argument('--geo_weight', type=float, default=1e-3, help='controls the influence of geometry loss')
    parser.add_argument('--checkpoint_path', type=str,
                        default=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\checkpoints\forward.pth')
    parser.add_argument('--patience', type=int, default=7, help='early stopping patience')
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pca = pickle.load(open(os.path.join(args.data_path, 'pca_model.pkl'), 'rb'))
    antenna_dataset_loader = AntennaDataSetLoader(args.data_path, batch_size=args.batch_size, pca=pca)
    model = forward_GammaRad(radiation_channels=12)
    loss_fn = GammaRad_loss(geo_weight=args.geo_weight)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma_schedule)
    keep_training = True
    epoch = 0
    patience = args.patience
    best_loss = np.inf
    train_loss = 0
    if os.path.exists(os.path.join(args.data_path, 'env_scaler.pkl')):
        env_scaler = pickle.load(open(os.path.join(args.data_path, 'env_scaler.pkl'), 'rb'))
    else:
        print('Fitting environment scaler.')
        env_scaler = standard_scaler()
        envs = []
        for i, X in enumerate(antenna_dataset_loader.trn_loader):
            print(f'Loaded {i} batches for environment scaler') if i % 10 == 0 else None
            _, _, _, env = X
            envs.append(env)
        envs = torch.cat(envs, dim=0).detach().cpu().numpy()
        env_scaler.fit(envs)
        pickle.dump(env_scaler, open(os.path.join(args.data_path, 'env_scaler.pkl'), 'wb'))
        print(f'Environment scaler fitted and saved in: {os.path.join(args.data_path, "env_scaler.pkl")}')
    while keep_training:
        if epoch % 10 == 0 and epoch > 0:
            print(f'Saving model at epoch {epoch}')
            torch.save(model.state_dict(), args.checkpoint_path.replace('.pth', f'_epoch{epoch}.pth'))

        print(f'Starting Epoch: {epoch}. Patience: {patience}')
        model.train()
        for idx, sample in enumerate(antenna_dataset_loader.trn_loader):
            EMBEDDINGS, GAMMA, RADIATION, ENV = sample
            embeddings, gamma, radiation, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
                env_scaler.forward(ENV).to(device)
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
                print(f'Epoch: {epoch}, Batch: {idx}, Loss: {train_loss / 100}')
        print(f'End Epoch: {epoch}, Loss: {train_loss / len(antenna_dataset_loader.trn_loader)}')
        epoch += 1
        lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate: {lr}')
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for idx, sample in enumerate(antenna_dataset_loader.val_loader):
                EMBEDDINGS, GAMMA, RADIATION, ENV = sample
                embeddings, gamma, radiation, env = EMBEDDINGS.to(device), GAMMA.to(device), RADIATION.to(device), \
                    env_scaler.forward(ENV).to(device)
                geometry = torch.cat((embeddings, env), dim=1)
                target = (gamma, radiation)
                gamma_pred, rad_pred = model(geometry)
                output = (gamma_pred, rad_pred, geometry)
                loss = loss_fn(output, target)
                val_loss += loss.item()
            print(f'Validation Loss: {val_loss / len(antenna_dataset_loader.val_loader)}')
            if train_loss < best_loss:
                best_loss = train_loss
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
    torch.save(best_model.state_dict(), args.checkpoint_path.replace('.pth', '_best_dict.pth'))
    torch.save(best_model, args.checkpoint_path.replace('.pth', '_best_instance.pth'))
    print('Training finished.')
    print(f'Best loss: {best_loss}')
    print(f'Best model saved at: {args.checkpoint_path}')
