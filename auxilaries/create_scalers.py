from AntennaDesign.utils import *
import os

data_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\model_6\processed_data'
antenna_dataset_loader = AntennaDataSetsLoader(data_path, split_ratio=[0.9, 0.1, 0.])
ant_scaler, env_scaler = standard_scaler(), standard_scaler()
ant_scaler_path = os.path.join(data_path, 'ant_scaler.pkl')
ant_scaler_manager = ScalerManager(ant_scaler_path, ant_scaler)
env_scaler_path = os.path.join(data_path, 'env_scaler.pkl')
env_scaler_manager = ScalerManager(env_scaler_path, env_scaler)

for idx, (embeddings, gamma, radiation, env, name) in enumerate(antenna_dataset_loader.trn_loader):
    ant_scaler_manager.fit_and_dump(embeddings.numpy())
    env_scaler_manager.fit_and_dump(env.numpy())
    print('DONE')
