import numpy as np
import argparse
import pytorch_msssim
import utils
from utils import standard_scaler, create_dataset, split_dataset, create_dataloader
from models import baseline_regressor, inverse_hypernet, forward_radiation,forward_GammaRad, inverse_transformer,forward_gamma
import torch
import trainer
import matplotlib.pyplot as plt
import pickle
from losses import *
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'../AntennaDesign_data/newdata_dB.npz')
    parser.add_argument('--forward_model_path_gamma', type=str, default=r'checkpoints/forward_gamma_smoothness_0.001_0.0001.pth')
    parser.add_argument('--forward_model_path_radiation', type=str, default=r'checkpoints/forward_radiation_huberloss.pth')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--gamma_schedule', type=float, default=0.95, help='gamma decay rate')
    parser.add_argument('--step_size', type=int, default=9, help='step size for gamma decay')
    parser.add_argument('--grad_accumulation_step', type=int, default=None, help='gradient accumulation step. Should be None if HyperNet is not used')
    parser.add_argument('--inv_or_forw', type=str, default='inverse_forward_GammaRad',
    help='architecture name, to parse dataset correctly. options: inverse, forward_gamma, forward_radiation, inverse_forward_gamma, inverse_forward_GammaRad')
    parser.add_argument('--rad_range', type=list, default=[-55,5], help='range of radiation values for scaling')
    parser.add_argument('--GammaRad_lambda', type=float, default=1.0, help='controls the influence of radiation in GammaRad loss')
    parser.add_argument('--rad_phase_factor', type=float, default=1.0, help='controls the influence of radiations phase in GammaRad loss')
    parser.add_argument('--mag_smooth_weight', type=float, default=1e-3, help='controls the influence of gamma magnitude smoothness')
    parser.add_argument('--phase_smooth_weight', type=float, default=1e-3, help='controls the influence of gamma phase smoothness')
    parser.add_argument('--geo_weight', type=float, default=1e-3, help='controls the influence of geometry loss')
    parser.add_argument('--checkpoint_path', type=str, default=r'checkpoints/inverseforward_bigger_data.pth')
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
    # ---------------------
    batch_size, epochs = args.batch_size, args.epochs
    radiation_range = args.rad_range
    learning_rates = args.lr
    grad_accum_stp = args.grad_accumulation_step
    stp_size,gamma_schedule = args.step_size,args.gamma_schedule
    inv_or_forw = args.inv_or_forw
    GammaRad_lambda,rad_phase_fac = args.GammaRad_lambda,args.rad_phase_factor
    path = args.checkpoint_path
    g_weight = args.geo_weight
    #----------------------
    print('bs=', batch_size, ' step_size=', stp_size)

    model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_hypernet.small_inverse_radiation_no_hyper(),
                                             forw_module=forward_GammaRad.forward_GammaRad(rad_range=radiation_range),
                                             forward_weights_path_rad=args.forward_model_path_radiation,
                                             forward_weights_path_gamma=args.forward_model_path_gamma)
    loss_fn = GammaRad_loss(lamda=GammaRad_lambda,rad_phase_fac=rad_phase_fac,geo_weight=g_weight)
    # model = forward_radiation.Radiation_Generator(radiation_range)
    # loss_fn = radiation_loss_dB(mag_loss='huber',rad_phase_factor=rad_phase_fac)
    #loss_fn = gamma_loss_dB(mag_smooth_weight=args.mag_smooth_weight,phase_smooth_weight=args.phase_smooth_weight)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)
    train_loader = create_dataloader(train_gamma, train_radiation, train_params_scaled, batch_size, device,inv_or_forw)
    val_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device,inv_or_forw)
    test_loader = create_dataloader(test_gamma, test_radiation, test_params_scaled, batch_size, device,inv_or_forw)
    print(f'seccessfully created data loaders for {inv_or_forw} training')
    train_los, train_los_stds, val_los, val_los_stds,test_loss,best_state_dict,best_val_loss = trainer.run_model(model, loss_fn, optimizer,
                                                                 train_loader, val_loader, test_loader,epochs, stp_size,
                                                                 gamma_schedule,inv_or_forw,
                                                                 grad_accumulation_step=grad_accum_stp)
    print('best model saved for validation loss = ', best_val_loss)
    print('saving best model')
    torch.save(best_state_dict, path)
    args_path = path[:-4] + '_args.txt'
    with open(args_path, 'w') as args_file:
        args_file.write(str(args))
        args_file.write('\n')
        args_file.write(f'best validation loss = {best_val_loss}')
    plt.figure()
    plt.plot(np.arange(len(val_los)), val_los, label='validation')
    plt.plot(np.arange(len(val_los)), train_los, label='train')
    #plt.plot(np.arange(len(val_los)), np.ones(len(val_los)) * nn_loss.item(), label='NN loss', color='k')
    plt.title(f'model test loss = {test_loss:.3f}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.figure()
    plt.plot(np.arange(len(val_los)), val_los_stds, label='validation')
    plt.plot(np.arange(len(val_los)), train_los_stds, label='train')
    plt.title(f'data')
    plt.xlabel('epoch')
    plt.ylabel('std')
    plt.legend()
    plt.show()