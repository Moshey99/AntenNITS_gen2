import os
import re
from AntennaDesign.utils import *
import torch
from forward_model_evaluate_main import plot_condition
import matplotlib.pyplot as plt


def remove_filter_tag(filter_tag: str):
    to_remove = ''
    if filter_tag.__contains__('nn'):
        to_remove = r'_nn'
    elif filter_tag.__contains__('grade'):
        to_remove = r'_grade_\d+'
    elif filter_tag.__contains__('shahar'):
        to_remove = r'_shahar'

    return to_remove


if __name__ == "__main__":
    filter_tag = 'grade_0'  # can be also 'nn' or 'grade_0'
    all_gamma_stats, all_radiation_stats = [], []
    total_counter = 0
    gamma_only_improve_counter = 0
    rad_only_improve_counter = 0
    both_improve_counter = 0
    both_big_improve_counter = 0
    both_worse_counter = 0
    gamma_th, rad_th = 2.89, 2.59
    # cst_folder = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_cst_results_dipole"
    cst_folder = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\all_logs_generated_samples_val_data_checkpoint_directivity_mag_only\processed_results"
    data_path = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_data_130k_200k"
    visited_antennas = []
    cst_antenna_folders = [os.path.join(cst_folder, folder) for folder in os.listdir(cst_folder)]
    cst_folders = [folder for folder in cst_antenna_folders if filter_tag in os.path.basename(folder)]
    rad_stats_buffer = []
    gamma_stats_buffer = []
    # s = {'SPEC_dipole_z_aligned_ENV_phone_3', 'SPEC_dipole_z_aligned_ENV_phone_4', 'SPEC_dipole_z_aligned_ENV_phone_bars_3',
    #      'SPEC_dipole_z_aligned_ENV_phone_reflector_2', 'SPEC_dipole_z_aligned_ENV_phone_reflector_3',
    #      'SPEC_dual_dipole_with_ground_z_aligned_ENV_196744', 'SPEC_dual_dipole_z_aligned_ENV_196744'}
    # pattern = 'z_aligned'
    for folder in cst_folders:
        # if not folder.__contains__(pattern) or os.path.basename(folder)[:-8] in s:
        #     continue
        cst_gam = downsample_gamma(np.load(os.path.join(folder, 'gamma.npy'))[np.newaxis], rate=4).squeeze()
        cst_gam = torch.tensor(AntennaDataSet.clip_gamma(cst_gam)).float()
        cst_rad = downsample_radiation(np.load(os.path.join(folder, 'radiation.npy'))[np.newaxis],
                                       rates=[4, 2]).squeeze()
        cst_rad = torch.tensor(AntennaDataSet.clip_radiation(cst_rad)).float()
        cst_name = os.path.basename(folder)
        addon_to_remove = remove_filter_tag(filter_tag)

        antenna_name = re.sub(addon_to_remove, '', cst_name)
        print('Working on CST antenna:', cst_name, 'Matching test case:', antenna_name)
        gt_gam = downsample_gamma(np.load(os.path.join(data_path, antenna_name, 'gamma.npy'))[np.newaxis],
                                  rate=4).squeeze()
        gt_gam = torch.tensor(AntennaDataSet.clip_gamma(gt_gam)).float()
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
        if antenna_name not in visited_antennas:
            visited_antennas.append(antenna_name)
            if len(gamma_stats_buffer) > 0:
                best_gamma_idx_in_buffer = torch.stack(gamma_stats_buffer)[:, 0].argmin()
                best_rad_idx_in_buffer = torch.stack(rad_stats_buffer)[:, 0].argmin()
                best_gamma_stats = gamma_stats_buffer[best_gamma_idx_in_buffer]
                best_rad_stats = rad_stats_buffer[best_rad_idx_in_buffer]
                gamma_stats_buffer = []
                rad_stats_buffer = []
                all_gamma_stats.append(best_gamma_stats)
                all_radiation_stats.append(best_rad_stats)
                total_counter += 1
                if best_gamma_stats[0] < gamma_th:
                    gamma_only_improve_counter += 1
                if best_rad_stats[0] < rad_th:
                    rad_only_improve_counter += 1
                if best_gamma_stats[0] < gamma_th and best_rad_stats[0] < rad_th:
                    both_improve_counter += 1
                if best_gamma_stats[0] < gamma_th / 2 or best_rad_stats[0] < rad_th / 2:
                    both_big_improve_counter += 1
                if best_gamma_stats[0] > gamma_th and best_rad_stats[0] > rad_th:
                    both_worse_counter += 1
        gamma_stats_buffer.append(gamma_stats)
        rad_stats_buffer.append(rad_stats)
    if len(gamma_stats_buffer) > 0:
        best_gamma_idx_in_buffer = torch.stack(gamma_stats_buffer)[:, 0].argmin()
        best_rad_idx_in_buffer = torch.stack(rad_stats_buffer)[:, 0].argmin()
        best_gamma_stats = gamma_stats_buffer[best_gamma_idx_in_buffer]
        best_rad_stats = rad_stats_buffer[best_rad_idx_in_buffer]
        gamma_stats_buffer = []
        rad_stats_buffer = []
        all_gamma_stats.append(best_gamma_stats)
        all_radiation_stats.append(best_rad_stats)
        total_counter += 1
        if best_gamma_stats[0] < gamma_th:
            gamma_only_improve_counter += 1
        if best_rad_stats[0] < rad_th:
            rad_only_improve_counter += 1
        if best_gamma_stats[0] < gamma_th and best_rad_stats[0] < rad_th:
            both_improve_counter += 1
        if best_gamma_stats[0] < gamma_th / 2 or best_rad_stats[0] < rad_th / 2:
            both_big_improve_counter += 1
        if best_gamma_stats[0] > gamma_th and best_rad_stats[0] > rad_th:
            both_worse_counter += 1

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
