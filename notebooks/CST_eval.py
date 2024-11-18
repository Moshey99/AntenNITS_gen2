import os
from AntennaDesign.utils import *
import torch
from forward_model_evaluate_main import plot_condition
import matplotlib.pyplot as plt


if __name__ == "__main__":
    filter_tag = 'nn'  # can be also 'nn' or 'grade_0'
    all_gamma_stats, all_radiation_stats = [], []
    total_counter = 0
    gamma_only_improve_counter = 0
    rad_only_improve_counter = 0
    both_improve_counter = 0
    both_big_improve_counter = 0
    both_worse_counter = 0
    gamma_th, rad_th = 0.66, 3.05
    cst_folder = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_110k_150k_processed\CST_results_processed'
    data_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_110k_150k_processed'
    cst_antenna_folders = [os.path.join(cst_folder, folder) for folder in os.listdir(cst_folder)]
    cst_folders = [folder for folder in cst_antenna_folders if filter_tag in os.path.basename(folder)]
    for folder in cst_folders:
        cst_gam = downsample_gamma(np.load(os.path.join(folder, 'gamma.npy'))[np.newaxis], rate=4).squeeze()
        cst_gam = torch.tensor(cst_gam).float()
        cst_rad = downsample_radiation(np.load(os.path.join(folder, 'radiation.npy'))[np.newaxis],
                                       rates=[4, 2]).squeeze()
        cst_rad = torch.tensor(AntennaDataSet.clip_radiation(cst_rad)).float()
        cst_name = os.path.basename(folder)
        antenna_name = cst_name.split('_')[0]
        print('Working on CST antenna:', cst_name)
        gt_gam = downsample_gamma(np.load(os.path.join(data_path, antenna_name, 'gamma.npy'))[np.newaxis],
                                  rate=4).squeeze()
        gt_gam = torch.tensor(gt_gam).float()
        gt_rad = downsample_radiation(np.load(os.path.join(data_path, antenna_name, 'radiation.npy'))[np.newaxis],
                                      rates=[4, 2]).squeeze()
        gt_rad = torch.tensor(AntennaDataSet.clip_radiation(gt_rad))
        cst_gam, cst_rad = cst_gam.unsqueeze(0), cst_rad.unsqueeze(0)
        gt_gam, gt_rad = gt_gam.unsqueeze(0), gt_rad.unsqueeze(0)
        # x=2
        # cst_rad = torch.cat((cst_rad[:,x:x+2], cst_rad[:,x+6:x+8]), dim=1)
        # gt_rad = torch.cat((gt_rad[:,x:x+2], gt_rad[:,x+6:x+8]), dim=1)
        gamma_stats = torch.tensor(produce_gamma_stats(gt_gam, cst_gam, dataset_type='dB', to_print=True))
        rad_stats = torch.tensor(produce_radiation_stats(gt_rad, cst_rad, to_print=True))
        all_gamma_stats.append(gamma_stats)
        all_radiation_stats.append(rad_stats)
        total_counter += 1
        if gamma_stats[0] < gamma_th:
            gamma_only_improve_counter += 1
        if rad_stats[0] < rad_th:
            rad_only_improve_counter += 1
        if gamma_stats[0] < gamma_th and rad_stats[0] < rad_th:
            both_improve_counter += 1
        if gamma_stats[0] < gamma_th/2 and rad_stats[0] < rad_th/2:
            both_big_improve_counter += 1
        if gamma_stats[0] > gamma_th and rad_stats[0] > rad_th:
            both_worse_counter += 1
        # plot_condition((cst_gam, cst_rad), freqs=np.arange(cst_gam.shape[1] // 2))
        # plot_condition((gt_gam, gt_rad), freqs=np.arange(gt_gam.shape[1] // 2))
        # plt.show()
    print(f'Total antennas: {total_counter}')
    print(f'Gamma only improvement: {gamma_only_improve_counter}')
    print(f'Radiation only improvement: {rad_only_improve_counter}')
    print(f'Both improved: {both_improve_counter}')
    print(f'Both improved by a lot: {both_big_improve_counter}')
    print(f'Both worse: {both_worse_counter}')
    mean_gamma_stats, _ = torch.median(torch.stack(all_gamma_stats), dim=0)
    mean_rad_stats, _ = torch.median(torch.stack(all_radiation_stats), dim=0)
    print('Mean gamma stats:', mean_gamma_stats)
    print('Mean radiation stats:', mean_rad_stats)