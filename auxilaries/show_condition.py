from AntennaDesign.utils import *
import os
import torch
from notebooks.forward_model_evaluate_main import plot_condition

data_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\model_6\processed_data'
antenna_dataset_loader = AntennaDataSetsLoader(data_path, split_ratio=[0.9, 0.1, 0.], batch_size=1)
test_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\model_6\processed_reference_sweep'
antenna_dataset_loader.load_test_data(test_path)

for idx, (_, GAMMA, RADIATION, _, name) in enumerate(antenna_dataset_loader.tst_loader):
    device = torch.device("cpu")
    gamma, rad = GAMMA.to(device), RADIATION.to(device)
    print('Working on antenna: ', name[0])
    plot_condition((gamma, rad), title=name[0])
    plt.show()
    print('DONE')