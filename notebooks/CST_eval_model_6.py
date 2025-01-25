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
        #OBWO1 = OBWO1*FBW1
    else:
        FBW1, OBWO1 = 0, 0

    if len(frequencies_below_threshold_f2) > 0:
        BW2 = [min(frequencies_below_threshold_f2), max(frequencies_below_threshold_f2)]
        fc2 = frequencies_below_threshold_f2[np.argmin(gamma[np.where((gamma < threshold) & (frequencies > 4000))[0]])]
        FBW2 = (max(BW2)-min(BW2))/fc2
        overlap2 = overlap(BW2, optimal_bw2)
        OBWO2 = (max(overlap2)-min(overlap2))/(max(optimal_bw2)-min(optimal_bw2)) if overlap2 != 0 else 0
        #OBWO2 = OBWO2*FBW2
    else:
        FBW2, OBWO2 = 0, 0
    print(f'FBW1: {np.round(100*FBW1,2)}, OBWO1: {np.round(100*OBWO1,2)}, FBW2: {np.round(100*FBW2,2)}, OBWO2: {np.round(100*OBWO2,2)}')
    return FBW1*100, OBWO1, FBW2*100, OBWO2


if __name__ == "__main__":
    filter_tag = ''  # can be also 'nn' or 'grade_0'
    all_gamma_stats = []
    all_gammas = []
    # cst_folder = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\processed_cst_results_dipole"
    cst_folder = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\model_6\all_logs_generated_sweep_testdata_NN\results"
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


    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def calc_effective_bfw(bfw,obwo_percentage):
        obwo_frac = np.array(obwo_percentage) / 100
        return np.array(bfw)*obwo_frac

    test_cases = [20, 30, 40, 50, 60]

    ours_fbw1 = [0,10.85,19.08,10.14,5.98]
    ours_obwo1 = [0,67.95,100,59.59,34.86]
    ours_fbw2 = [16.84,17.83,20.17,15.68,16.23]
    ours_obwo2 = [100]*5
    ours_efbw1 = np.round(calc_effective_bfw(ours_fbw1,ours_obwo1),2)
    ours_efbw2 = np.round(calc_effective_bfw(ours_fbw2,ours_obwo2),2)

    theirs_fbw1 = [0,9.27,17.80,9.12,5.13]
    theirs_obwo1 = [0,57.21,100,44.85,33.47]
    theirs_fbw2 = [14.14,14.54,18.05,14.39,16.11]
    theirs_obwo2 = [98.59,100,100,100,100]
    theirs_efbw1 = np.round(calc_effective_bfw(theirs_fbw1, theirs_obwo1),2)
    theirs_efbw2 = np.round(calc_effective_bfw(theirs_fbw2, theirs_obwo2),2)

    nn_fbw1 = [0,0,34.52,16.24,0]
    nn_obwo1 = [0,0,100,85.82,0]
    nn_obwo2 = [0,24.68,5.58,3.88,0]
    nn_fbw2 = [0,24.22,45.25,30.83,0]
    nn_efbw1 = np.round(calc_effective_bfw(nn_fbw1, nn_obwo1),2)
    nn_efbw2 = np.round(calc_effective_bfw(nn_fbw2, nn_obwo2),2)
    # plt.figure()
    # y2_ours = [16.84,17.83,20.17,15.68,16.23]
    # y2_theirs = [14.14,14.54,18.05,14.39,16.11]
    # plt.scatter(x, y2_ours, label='Ours', color='r',alpha=0.5)
    # plt.scatter(x,y2_theirs, label='Theirs',color='b',alpha=0.5)
    # plt.xlabel('Lg')
    # plt.xticks([20, 30, 40, 50, 60])  # Set the x-axis ticks
    #
    # plt.ylabel('FBW(%)')
    # plt.title('FBW-f2')
    #
    # plt.legend()
    # plt.show()

    # Create the bar chart
    width = 0.2  # Width of the bars
    x = np.arange(len(test_cases))

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 3, ours_efbw1, width, label='Ours', color='blue')
    rects2 = ax.bar(x - width / 3 + width, theirs_efbw1, width, label='Theirs', color='orange')
    rects3 = ax.bar(x - width / 3 + 2*width, nn_efbw1, width, label='NN', color='green')



    # Add labels, title, and legend
    ax.set_ylabel('eFBW-f1 (%)')
    ax.set_xlabel('Lg Test Case')
    ax.set_title('Comparison of eFBW-f1: Ours vs. Theirs')
    ax.set_xticks(x)
    ax.set_xticklabels(test_cases)
    ax.legend()

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.ylim(0, 50)  # Adjust y-axis limits for better visualization
#-----------------------------------------------------------------------------


    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 3, ours_efbw2, width, label='Ours', color='blue')
    rects2 = ax.bar(x - width / 3 + width, theirs_efbw2, width, label='Theirs', color='orange')
    rects3 = ax.bar(x - width / 3 + 2*width, nn_efbw2, width, label='NN', color='green')

    # Add labels, title, and legend
    ax.set_ylabel('eFBW-f2 (%)')
    ax.set_xlabel('Lg Test Case')
    ax.set_title('Comparison of eFBW-f2: Ours vs. Theirs')
    ax.set_xticks(x)
    ax.set_xticklabels(test_cases)
    ax.legend()

    add_labels(rects1)
    add_labels(rects2)
    add_labels(rects3)

    plt.ylim(0, 52)  # Adjust y-axis limits for better visualization
#-----------------------------------------------------------------------------------------------
#     test_cases = [20, 30, 40, 50, 60]
#     theirs_obwo2 = [0,57.21,100,44.85,33.47]
#     ours_obwo2 = [0,67.95,100,59.59,34.86]
#
#     # Create the bar chart
#     width = 0.35  # Width of the bars
#     x = np.arange(len(test_cases))
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width / 2, ours_obwo2, width, label='Ours', color='blue')
#     rects2 = ax.bar(x + width / 2, theirs_obwo2, width, label='Theirs', color='orange')
#
#     # Add labels, title, and legend
#     ax.set_ylabel('OBWO-f1 (%)')
#     ax.set_xlabel('Lg Test Case')
#     ax.set_title('Comparison of OBWO-f1: Ours vs. Theirs')
#     ax.set_xticks(x)
#     ax.set_xticklabels(test_cases)
#     ax.legend()
#
#     add_labels(rects1)
#     add_labels(rects2)
#
#     plt.ylim(0, 110)  # Adjust y-axis limits for better visualization
# #-----------------------------------------------------------------------------------------------
#     # Sample data (replace with your actual data)
#     test_cases = [20, 30, 40, 50, 60]
#     ours_obwo2 = [100.0, 100.0, 100.0, 100.0, 100.0]
#     theirs_obwo2 = [98.59, 100.0, 100.0, 100.0, 100.0]
#
#     # Create the bar chart
#     width = 0.35  # Width of the bars
#     x = np.arange(len(test_cases))
#
#     fig, ax = plt.subplots()
#     rects1 = ax.bar(x - width/2, ours_obwo2, width, label='Ours', color='blue')
#     rects2 = ax.bar(x + width/2, theirs_obwo2, width, label='Theirs', color='orange')
#
#     # Add labels, title, and legend
#     ax.set_ylabel('OBWO-f2 (%)')
#     ax.set_xlabel('Lg Test Case')
#     ax.set_title('Comparison of OBWO-f2: Ours vs. Theirs')
#     ax.set_xticks(x)
#     ax.set_xticklabels(test_cases)
#     ax.legend()
#
#     add_labels(rects1)
#     add_labels(rects2)
#
#     plt.ylim(95, 102)  # Adjust y-axis limits for better visualization
    plt.tight_layout()
    plt.show()
