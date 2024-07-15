import matplotlib.pyplot as plt
from utils import *
import trainer
from models import baseline_regressor, inverse_hypernet, forward_radiation,forward_GammaRad,inverse_transformer
from losses import *
import torch
import argparse
import scipy.io as sio

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'../AntennaDesign_data/newdata_dB.npz')
    parser.add_argument('--model_path', type=str, default='checkpoints/forward_radiation_huberloss.pth')
    parser.add_argument('--inv_or_forw', type=str, default='forward_radiation',
    help='architecture name, to parse dataset correctly. options: inverse, forward_gamma, forward_radiation, inverse_forward_gamma, inverse_forward_GammaRad')
    parser.add_argument('--sample', type=int, default=180, help='sample to plot its output, from test set')


    return parser.parse_args()

def preprocess_mat(mat):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = np.squeeze(mat['gamma'])
    gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
    rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
    rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
    rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
    gamma,radiation = torch.tensor(gamma).to(dev).unsqueeze(0),torch.tensor(rad_concat_swapped).to(dev).unsqueeze(0)
    gamma_down = downsample_gamma(gamma,4).squeeze(0)
    radiation_down = downsample_radiation(radiation, rates=[4, 2]).squeeze(0)
    gamma_down[:gamma_down.shape[0]//2] = 10*np.log10(gamma_down[:gamma_down.shape[0]//2])
    radiation_down[:radiation_down.shape[0]//2] = 10*np.log10(radiation_down[:radiation_down.shape[0]//2])
    return gamma_down,radiation_down.cpu().detach().numpy()
def main():
    #---------------
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data_path)
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    inv_or_forw = args.inv_or_forw
    sample = args.sample
    #---------------
    model = forward_radiation.Radiation_Generator([-55,5])
    loss_fn = None
    model.to(device)
    model.load_state_dict(torch.load(args.model_path,map_location=device))
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)
    batch_size = val_params_scaled.shape[0]
    test_loader = create_dataloader(val_gamma, val_radiation,val_params_scaled, batch_size, device, inv_or_forw)
    predicted_spectrums,gt_spectrums = trainer.evaluate_model(model, loss_fn, test_loader, 'test', inv_or_forw, return_output=True)
    #---

    pred_radiation = predicted_spectrums
    GT_radiation = gt_spectrums
    produce_radiation_stats(gt_spectrums, predicted_spectrums)
    prnt = inv_or_forw
    plt.figure()
    plt.imshow(pred_radiation[sample, 2, :, :].cpu().detach().numpy())
    plt.title(f'Predicted radiation pattern phase, {prnt} loss, sample {args.sample}')
    plt.colorbar()
    plt.figure()
    plt.imshow(GT_radiation[sample, 2, :, :].cpu().detach().numpy())
    plt.title(f'Ground truth radiation pattern phase, sample {args.sample}')
    plt.colorbar()
    plt.figure()
    plt.imshow(pred_radiation[sample, 0, :, :].cpu().detach().numpy())
    plt.title(f'Predicted radiation pattern magnitude, {prnt} loss, sample {args.sample}')
    plt.colorbar()
    plt.figure()
    plt.imshow(GT_radiation[sample, 0, :, :].cpu().detach().numpy())
    plt.title(f'Ground truth radiation pattern magnitude, sample {args.sample}')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()