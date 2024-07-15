import scipy.io as sio
import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
import utils
from utils import standard_scaler, create_dataset, split_dataset,create_dataloader
from models import baseline_regressor, inverse_hypernet
import torch
import trainer
import matplotlib.pyplot as plt
import pickle
from losses import *

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'../AntennaDesign_data/data_dB.npz')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--gamma_schedule', type=float, default=0.97, help='gamma decay rate')
    parser.add_argument('--step_size', type=int, default=2, help='step size for gamma decay')
    parser.add_argument('--grad_accumulation_step', type=int, default=5, help='gradient accumulation step. Should be None if HyperNet is not used')
    parser.add_argument('--inv_or_forw', type=str, default='inverse',
    help='architecture name, to parse dataset correctly. options: inverse, forward_gamma, forward_radiation, inverse_forward_gamma, inverse_forward_GammaRad')
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data_path)
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']

    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)
    #------------------------------
    batch_size, epochs = args.batch_size, args.epochs
    grad_accum_stp = args.grad_accumulation_step
    learning_rates = args.lr
    stp_size,gamma_schedule = args.step_size, args.gamma_schedule
    inv_or_forw = args.inv_or_forw
    #------------------------------
    print('bs=', batch_size, ' step_size=', stp_size)
    model = inverse_hypernet.inverse_radiation_hyper()
    loss_fn = nn.HuberLoss()
    # model = baseline_regressor.baseline_inverse_model()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)
    train_loader = create_dataloader(train_gamma, train_radiation, train_params_scaled, batch_size, device,inv_or_forw)
    val_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device,inv_or_forw)
    test_loader = create_dataloader(test_gamma, test_radiation, test_params_scaled, batch_size, device,inv_or_forw)
    print(f'seccessfully created data loaders for {inv_or_forw} training')
    train_los, train_los_stds, val_los, val_los_stds,test_loss = trainer.run_model(model, loss_fn, optimizer,
    train_loader, val_loader, test_loader,epochs, stp_size, gamma_schedule,inv_or_forw,grad_accumulation_step=grad_accum_stp)


    plt.figure()
    plt.plot(np.arange(len(val_los)), val_los, label='validation')
    plt.plot(np.arange(len(val_los)), train_los, label='train')
    #plt.plot(np.arange(len(val_los)), np.ones(len(val_los)) * nn_loss.item(), label='NN loss', color='k')
    plt.title(f'loss for {inv_or_forw} model')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(len(val_los)), val_los_stds, label='validation')
    plt.plot(np.arange(len(val_los)), train_los_stds, label='train')
    plt.title(f'std for {inv_or_forw} model')
    plt.xlabel('epoch')
    plt.ylabel('std')
    plt.legend()
    print('saving best model')
    #torch.save(best_model.state_dict(), f'FORWARD_small_10layers_dB.pth')
    plt.show()