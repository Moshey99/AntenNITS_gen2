import matplotlib.pyplot as plt
from utils import *
import trainer
from models import baseline_regressor, inverse_hypernet, forward_radiation, forward_GammaRad, inverse_transformer
from losses import *
import torch
import argparse
import scipy.io as sio


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=r'../AntennaDesign_data/newdata_dB.npz')
    parser.add_argument('--model_path', type=str, default='checkpoints/inverseforward_bigger_data.pth')
    parser.add_argument('--inv_or_forw', type=str, default='inverse_forward_GammaRad',
                        help='architecture name, to parse dataset correctly. options: inverse, forward_gamma, forward_radiation, inverse_forward_gamma, inverse_forward_GammaRad')
    parser.add_argument('--rad_range', type=list, default=[-55, 5], help='range of radiation values [dB] for scaling')
    parser.add_argument('--sample', type=int, default=180, help='sample to plot its output, from test set')
    parser.add_argument('--compare_forward', type=bool, default=False,
                        help='compare forward output of predicted geo to cst file')

    return parser.parse_args()


def preprocess_mat(mat):
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gamma = np.squeeze(mat['gamma'])
    gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
    rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
    rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
    rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
    gamma, radiation = torch.tensor(gamma).to(dev).unsqueeze(0), torch.tensor(rad_concat_swapped).to(dev).unsqueeze(0)
    gamma_down = downsample_gamma(gamma, 4).squeeze(0)
    radiation_down = downsample_radiation(radiation, rates=[4, 2]).squeeze(0)
    gamma_down[:gamma_down.shape[0] // 2] = 20 * np.log10(gamma_down[:gamma_down.shape[0] // 2])
    radiation_down[:radiation_down.shape[0] // 2] = 10 * np.log10(radiation_down[:radiation_down.shape[0] // 2])
    return gamma_down, radiation_down.cpu().detach().numpy()


def main():
    # ---------------
    args = arg_parser()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data_path)
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    radiation_range = args.rad_range
    inv_or_forw = args.inv_or_forw
    sample = args.sample
    # ---------------
    # model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_transformer.inverse_transformer(),
    #                                               forw_module=forward_GammaRad.forward_GammaRad(rad_range=radiation_range))
    # loss_fn = GammaRad_loss(lamda=1,rad_phase_fac=1)
    model = inverse_hypernet.inverse_forward_concat(inv_module=inverse_hypernet.small_inverse_radiation_no_hyper(),
                                                    forw_module=forward_GammaRad.forward_GammaRad(
                                                        rad_range=radiation_range))
    loss_fn = GammaRad_loss(lamda=1, rad_phase_fac=1)
    # model = baseline_regressor.small_deeper_baseline_forward_model()
    # loss_fn = gamma_loss_dB(mag_smooth_weight=1e-3,phase_smooth_weight=0)
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    scaler = standard_scaler()
    scaler.fit(train_params)
    train_params_scaled = scaler.forward(train_params)
    val_params_scaled = scaler.forward(val_params)
    test_params_scaled = scaler.forward(test_params)
    batch_size = val_params_scaled.shape[0]
    test_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device, inv_or_forw)
    predicted_spectrums, gt_spectrums = trainer.evaluate_model(model, loss_fn, test_loader, 'test', inv_or_forw,
                                                               return_output=True)
    predicted_spectrums = predicted_spectrums[0], predicted_spectrums[1]
    test_loader = create_dataloader(val_gamma, val_radiation, val_params_scaled, batch_size, device, 'inverse')
    predicted_geo, gt_geo = trainer.evaluate_model(model.inverse_module, loss_fn, test_loader, 'test', 'inverse',
                                                   return_output=True)
    # ---
    predicted_geo_sample = scaler.inverse(predicted_geo[sample].cpu().detach().numpy())
    GT_geo_sample = scaler.inverse(gt_geo[sample].cpu().detach().numpy())
    print('gt geo scaled: ', gt_geo[sample])
    print('predicted geo scaled: ', predicted_geo[sample])
    pred_gamma, pred_radiation = predicted_spectrums
    GT_gamma, GT_radiation = gt_spectrums
    produce_gamma_stats(gt_spectrums, predicted_spectrums, 'dB')
    produce_radiation_stats(gt_spectrums, predicted_spectrums)
    print('--------gt vs predicted spectrum for predicted geo----------')
    if args.compare_forward:
        gt_forw_mat = sio.loadmat(r"C:\Users\moshey\Downloads\results_TEST_valdata_sample444_newforward.mat")
        gt_forw_gamma, gt_forw_radiation = preprocess_mat(gt_forw_mat)
        predicted_spectrum_sample = tuple([pred[sample:sample + 1] for pred in predicted_spectrums])
        produce_gamma_stats((torch.tensor(gt_forw_gamma).unsqueeze(0).to(device),
                             torch.tensor(gt_forw_radiation, dtype=torch.float32).unsqueeze(0).to(device))
                            , predicted_spectrum_sample, 'dB')
        produce_radiation_stats((torch.tensor(gt_forw_gamma).unsqueeze(0).to(device),
                                 torch.tensor(gt_forw_radiation, dtype=torch.float32).unsqueeze(0).to(device))
                                , predicted_spectrum_sample)
    prnt = inv_or_forw
    pred_gamma_sample = pred_gamma[sample].cpu().detach().numpy()
    pred_gamma_sample[:int(0.5 * GT_gamma.shape[1])] = 20 * np.log10(pred_gamma_sample[:int(0.5 * GT_gamma.shape[1])])
    GT_gamma_sample = GT_gamma[sample].cpu().detach().numpy()
    if args.compare_forward:
        plt.figure()
        plt.imshow(gt_forw_radiation[2, :, :])
        plt.title('ground truth radiation predicted geomtry pattern phase ')
        plt.colorbar()
        plt.figure()
        plt.imshow(gt_forw_radiation[0, :, :])
        plt.title('ground truth radiation predicted geomtry pattern magnitude ')
        plt.colorbar()
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
    plt.figure()
    plt.plot(pred_gamma_sample, label='Predicted gamma of predicted geo')
    plt.plot(GT_gamma_sample, label='GT gamma of GT geo')
    if args.compare_forward:
        plt.plot(gt_forw_gamma, label='GT gamma of predicted geo')
    plt.plot(np.ones(20) * 0.5 * GT_gamma.shape[1], np.arange(-1, 1, 0.1), 'k--')
    plt.title(f'Gamma, {prnt} loss, sample {args.sample}')
    plt.legend()
    print('GT geo: ', np.round(GT_geo_sample, 2))
    print('Predicted geo: ', np.round(predicted_geo_sample, 3))

    plt.show()


if __name__ == '__main__':
    main()
