import os
import re

import numpy as np

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


def interpolate_signal(signal, factor):
  original_size = len(signal)
  new_size = original_size * factor
  new_x = np.linspace(0, original_size - 1, new_size)
  interpolated_signal = np.interp(new_x, np.arange(original_size), signal)
  return interpolated_signal


def produce_model6_stats(gamma: np.array, frequencies: np.array):
    def overlap(range1, range2):
        if range1[0] > range2[1] or range1[1] < range2[0]:
            return 0
        max_overlap = min(range1[1], range2[1])
        min_overlap = max(range1[0], range2[0])
        return [min_overlap, max_overlap]

    gamma = interpolate_signal(gamma, factor=4)
    frequencies = interpolate_signal(frequencies, factor=4)
    optimal_bw1 = [2300, 2700]
    optimal_bw2 = [5150, 5850]
    threshold = -10
    frequencies_below_threshold = frequencies[gamma < threshold]
    frequencies_below_threshold_f1 = frequencies_below_threshold[frequencies_below_threshold < 4000]
    frequencies_below_threshold_f2 = frequencies_below_threshold[frequencies_below_threshold > 4000]
    if len(frequencies_below_threshold_f1) > 0:
        BW1 = [min(frequencies_below_threshold_f1), max(frequencies_below_threshold_f1)]
        fc1 = frequencies_below_threshold_f1[np.argmin(gamma[np.where((gamma < threshold) & (frequencies < 4000))[0]])]
        FBW1 = (max(BW1)-min(BW1))/fc1
        overlap1 = overlap(BW1, optimal_bw1)
        OBWO1 = (max(overlap1)-min(overlap1))/(max(optimal_bw1)-min(optimal_bw1)) if overlap1 != 0 else 0
    else:
        FBW1, OBWO1 = 0, 0

    if len(frequencies_below_threshold_f2) > 0:
        BW2 = [min(frequencies_below_threshold_f2), max(frequencies_below_threshold_f2)]
        fc2 = frequencies_below_threshold_f2[np.argmin(gamma[np.where((gamma < threshold) & (frequencies > 4000))[0]])]
        FBW2 = (max(BW2)-min(BW2))/fc2
        overlap2 = overlap(BW2, optimal_bw2)
        OBWO2 = (max(overlap2)-min(overlap2))/(max(optimal_bw2)-min(optimal_bw2)) if overlap2 != 0 else 0
    else:
        FBW2, OBWO2 = 0, 0
    print(f'FBW1: {np.round(100*FBW1,2)}, OBWO1: {np.round(100*OBWO1,2)}, FBW2: {np.round(100*FBW2,2)}, OBWO2: {np.round(100*OBWO2,2)}')
    return FBW1*100, OBWO1*100, FBW2*100, OBWO2*100


if __name__ == "__main__":
    filter_tag = ''  # can be also 'nn' or 'grade_0'
    all_gamma_stats = []
    all_gammas = []
    # cst_folder = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_cst_results_dipole"
    cst_folder = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\model_6\reference_sweep"
    visited_antennas = []
    cst_antenna_folders = [os.path.join(cst_folder, folder) for folder in os.listdir(cst_folder)]
    cst_folders = [folder for folder in cst_antenna_folders if filter_tag in os.path.basename(folder)]
    gamma_stats_buffer = []
    for folder in cst_folders:
        with open(os.path.join(folder, 'S_parameters.pickle'), 'rb') as file:
            gamma_raw, freqs = pickle.load(file)

        cst_gam = 20 * np.log10(np.abs(gamma_raw))
        cst_name = os.path.basename(folder)
        addon_to_remove = remove_filter_tag(filter_tag)
        all_gammas.append(cst_gam)
        antenna_name = re.sub(addon_to_remove, '', cst_name)
        print('Working on CST antenna:', cst_name, 'Matching test case:', antenna_name)
        gamma_stats = torch.tensor(produce_model6_stats(cst_gam, freqs))
        if antenna_name not in visited_antennas:
            visited_antennas.append(antenna_name)
            if len(gamma_stats_buffer) > 0:
                best_gamma_idx_in_buffer = torch.stack(gamma_stats_buffer)[:, 0].argmax()
                best_gamma_stats = gamma_stats_buffer[best_gamma_idx_in_buffer]
                gamma_stats_buffer = []
                all_gamma_stats.append(best_gamma_stats)

        gamma_stats_buffer.append(gamma_stats)
    if len(gamma_stats_buffer) > 0:
        best_gamma_idx_in_buffer = torch.stack(gamma_stats_buffer)[:, 0].argmax()
        best_gamma_stats = gamma_stats_buffer[best_gamma_idx_in_buffer]
        gamma_stats_buffer = []
        all_gamma_stats.append(best_gamma_stats)

    all_gamma_stats = torch.stack(all_gamma_stats)
    mean_gamma_stats = torch.mean(all_gamma_stats, dim=0)
    std_gamma_stats = torch.std(all_gamma_stats,dim=0)
    print_stats = [str(np.round(m.item(),2))+'+-'+str(np.round(s.item(),2)) for m, s in zip(mean_gamma_stats, std_gamma_stats)]
    print(f'gamma stats: {print_stats}')
    freqs /=1000
    for i,gamma in enumerate(all_gammas):
        plt.plot(freqs, gamma, label=f'LG = {20+10*i}mm')
    plt.plot(freqs, -10*np.ones_like(freqs), 'k--')
    plt.ylim([-25,0])
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Reflection Coefficient [dB]')
    plt.legend()
    plt.show()
