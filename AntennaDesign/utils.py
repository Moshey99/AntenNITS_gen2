import copy
import shutil
from typing import Union
import cv2
import scipy.io as sio
from scipy.ndimage import zoom
from os import listdir
import os
from pathlib import Path
from os.path import isfile, join
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
import trainer
import losses
from models import baseline_regressor, inverse_hypernet
import random
import pytorch_msssim
import glob
import pickle
import re
import open3d as o3d
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from shapely.geometry import Polygon


class DatasetPart:
    def __init__(self):
        self.x = None
        self.gamma = None
        self.radiation = None


class AntennaData:
    def __init__(self):
        self.Data = None
        self.n_dims = None
        self.trn = DatasetPart()
        self.val = DatasetPart()
        self.tst = DatasetPart()


class DataPreprocessor:
    def __init__(self, folder_path=None):
        if folder_path is not None:
            self.num_data_points = len(os.listdir(folder_path))
            self.folder_path = folder_path

    def environment_preprocessor(self, debug=False):
        # TODO: add the plane parameter to the environments
        print('Preprocessing environments')
        all_envs = []
        last_env = []
        folder_path = self.folder_path
        for i in sorted(os.listdir(folder_path)):
            print('working on environment number:', i, 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, i, 'model_parameters.pickle')
            try:
                with open(file_path, 'rb') as f:
                    env_dict = pickle.load(f)
                    env_dict.pop('type',
                                 None)  # remove the first element which is the type of the model (it's constant)
                    env_dict.pop('plane',
                                 None)  # remove the second element which is the plane of the model (not always exist)
                    env_vals = list(env_dict.values())
                    assert np.all([type(value) != list for value in env_vals]), 'ERROR. List in Environments values'
                    last_env = copy.deepcopy(env_vals)
                    all_envs.append(env_vals)
            except:
                print('Error in loading the file:', file_path)
                all_envs.append(last_env)
        all_envs = np.array(all_envs)
        if not debug:
            np.save(os.path.join(Path(folder_path).parent, 'processed_data', 'environments.npy'), all_envs)
        pass

    def radiation_preprocessor(self, plot=False, debug=False):
        print('Preprocessing radiations')
        all_radiations = []
        folder_path = self.folder_path
        for idx, i in enumerate(os.listdir(folder_path)):
            print('working on antenna number:', i, 'out of:', self.num_data_points)
            im_resized = np.zeros((4, 181, 91))
            file_path = os.path.join(folder_path, i, f'{i}_farfield.txt')
            df = pd.read_csv(file_path, sep='\s+', skiprows=[0, 1], header=None)
            df = df.apply(pd.to_numeric, errors='coerce')
            arr = np.asarray(df)
            angle_res = arr[1, 0] - arr[0, 0]
            angle1_res = int(180 / angle_res + 1)
            angle2_res = int(360 / angle_res)
            im = arr[:, 3:7]
            im_resh = np.transpose(im.reshape(angle2_res, angle1_res, -1), (2, 0, 1))
            im_resh = im_resh[[0, 2, 1, 3], :, :]  # rearrange the channels to be [mag1, mag2, phase1, phase2]
            for j in range(im_resh.shape[0]):
                current_im = im_resh[j]
                im_resized[j] = np.clip(cv2.resize(current_im, (91, 181), interpolation=cv2.INTER_LINEAR),
                                        current_im.min(), current_im.max())
                if plot:
                    titles = ['mag1', 'mag2', 'phase1', 'phase2']
                    plt.subplot(2, 2, j + 1)
                    if j < 2:
                        assert np.all(im_resized[j] >= 0), 'Negative values in radiation magnitude'
                        im_resized[j] = 10 * np.log10(im_resized[j])
                    else:
                        assert np.all(im_resized[j] >= 0) and np.all(
                            im_resized[j] <= 360), 'Phase values out of range 0-360'
                        im_resized[j] = np.deg2rad(im_resized[j]) - np.pi
                    plt.imshow(im_resized[j])
                    plt.title(titles[j])
                    plt.colorbar()
                    plt.show()
            all_radiations.append(im_resized)

        all_radiations = np.array(all_radiations)

        radiations_mag, radiations_phase = all_radiations[:, :2], all_radiations[:, 2:]
        assert np.all(radiations_mag >= 0), 'Negative values in radiation magnitude'
        assert np.all(radiations_phase >= 0) and np.all(radiations_phase <= 360), 'Phase values out of range 0-360'
        radiations_phase_radians = np.deg2rad(radiations_phase) - np.pi
        radiations_mag_dB = 10 * np.log10(radiations_mag)
        radiations = np.concatenate((radiations_mag_dB, radiations_phase_radians), axis=1)
        saving_folder = os.path.join(Path(folder_path).parent, 'processed_data')
        if not debug:
            np.save(os.path.join(saving_folder, 'radiations.npy'), radiations)
        print('Radiations saved successfully with mag in dB and phase in radians')

    def gamma_preprocessor(self, debug=False):
        print('Preprocessing gammas')
        all_gammas = []
        folder_path = self.folder_path
        for i in sorted(os.listdir(folder_path)):
            print('working on antenna number:', i, 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, i, f'{i}_S11.pickle')
            with open(file_path, 'rb') as f:
                gamma_raw = pickle.load(f)
                gamma_complex = gamma_raw[0]
                gamma_mag, gamma_phase = np.abs(gamma_complex), np.angle(
                    gamma_complex)  # gamma_phase in radians already
                assert np.all(gamma_mag >= 0), 'Negative values in gamma magnitude'
                assert np.all(gamma_phase >= -np.pi) and np.all(
                    gamma_phase <= np.pi), 'Phase values out of range -pi-pi'
                gamma_mag_dB = 10 * np.log10(gamma_mag)
                gamma = np.concatenate((gamma_mag_dB, gamma_phase))
                all_gammas.append(gamma)
        all_gammas = np.array(all_gammas)
        if not debug:
            np.save(os.path.join(Path(folder_path).parent, 'processed_data', 'gammas.npy'), all_gammas)
            np.save(os.path.join(Path(folder_path).parent, 'processed_data', 'frequencies.npy'), gamma_raw[1])
        print('Gammas saved successfully with mag in dB and phase in radians')
        pass

    @staticmethod
    def assert_radiation_rules(radiation: np.ndarray):
        eps = 1e-6
        mag_db = radiation[:, :int(radiation.shape[1] / 2)]
        phase_rad = radiation[:, int(radiation.shape[1] / 2):]
        mag_linear = 10 ** (mag_db / 10)
        assert np.all(mag_linear >= 0), 'Negative values in radiation magnitude'
        assert np.all(phase_rad >= -np.pi - eps) and np.all(
            phase_rad <= np.pi + eps), 'Phase values out of range -pi - pi radians'
        print('All radiation rules are satisfied!')
        return True

    @staticmethod
    def assert_gamma_rules(gamma: np.ndarray):
        eps = 1e-6
        mag_db = gamma[:, :int(gamma.shape[1] / 2)]
        phase_rad = gamma[:, int(gamma.shape[1] / 2):]
        mag_linear = 10 ** (mag_db / 10)
        assert np.all(mag_linear >= 0), 'Negative values in gamma magnitude'
        assert np.all(phase_rad >= -np.pi - eps) and np.all(
            phase_rad <= np.pi + eps), 'Phase values out of range -pi - pi radians'
        print('All gamma rules are satisfied!')
        return True

    def geometry_preprocessor(self):
        print('Preprocessing geometries')
        voxel_size = 0.125
        min_bound_org = np.array([5.4, 3.825, 6.375]) - voxel_size
        max_bound_org = np.array([54., 3.83, 42.5]) + voxel_size
        for i in range(2335, 10000):
            if i == 3234 or i == 9729:
                continue
            # a = np.load(r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_10000x1\data\processed_data\voxels\voxels_2050.npy')
            # plt.imshow(a[:,0,:].T)
            # plt.show()
            mesh = o3d.io.read_triangle_mesh(
                rf"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_10000x1\data\models\{i}\Antenna_PEC_STEP.stl")
            # o3d.visualization.draw_geometries([mesh])
            vg = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
                mesh, voxel_size=voxel_size, min_bound=min_bound_org, max_bound=max_bound_org)
            voxels = vg.get_voxels()
            indices = np.stack(list(vx.grid_index for vx in voxels))
            quary_x = np.arange(min_bound_org[0] + 0.5 * voxel_size, max_bound_org[0], step=voxel_size)
            quary_y = [3.825000047683716]
            quary_z = np.arange(min_bound_org[2] + 0.5 * voxel_size, max_bound_org[2], step=voxel_size)
            quary_array = np.zeros((len(quary_x), 1, len(quary_z)))
            start = time.time()
            for ii, x_val in enumerate(quary_x):
                for jj, y_val in enumerate(quary_y):
                    for kk, z_val in enumerate(quary_z):
                        ind = vg.get_voxel([x_val, y_val, z_val])
                        exists = np.any(np.all(indices == ind, axis=1))
                        quary_array[ii, jj, kk] = exists
            np.save(os.path.join(
                r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_10000x1\data\processed_data\voxels',
                f'voxels_{i}.npy'), quary_array.astype(bool))
            print(f'saved antenna {i}. Process time was:', time.time() - start)

    def stp_to_stl(self):
        import trimesh
        model_folder = 'C:\\Users\\moshey\\PycharmProjects\\etof_folder_git\\AntennaDesign_data\\data_10000x1\\data\\nits_checkpoints'
        all = []
        for i in range(10000):
            print('working on antenna number:', i + 1, 'out of:', self.num_data_points)
            stp_path = os.path.join(model_folder, f'{i}', 'Antenna_PEC_STEP.stp')
            mesh = trimesh.Trimesh(**trimesh.interfaces.gmsh.load_gmsh(
                file_name=stp_path, gmsh_args=[("Mesh.Algorithm", 1), ("Mesh.CharacteristicLengthFromCurvature", 1),
                                               ("General.NumThreads", 10),
                                               ("Mesh.MinimumCirclePoints", 32)]))
            mesh.export(os.path.join(model_folder, f'{i}', 'Antenna_PEC_STEP.stl'))
        print('Geometries saved successfully as an stl formated 3D triangle mesh')


class AntennaDataSet(torch.utils.data.Dataset):
    def __init__(self, antenna_folders: list[str], pca: PCA, try_cache: bool):
        self.antenna_folders = antenna_folders
        self.len = len(antenna_folders)
        self.pca = pca
        self.try_cache = try_cache
        self.antenna_hw = (144, 200)
        self.ant, self.embeddings, self.gam, self.rad, self.env = None, None, None, None, None

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        antenna_folder = self.antenna_folders[idx]
        self.load_antenna(antenna_folder)
        ant_resized = self.resize_antenna()
        if self.embeddings is None:
            self.embeddings = self.pca.transform(ant_resized.flatten().reshape(1, -1)).flatten()
        self.to_tensors()
        embs = self.embeddings.detach().clone()
        self.embeddings = None
        return embs, self.gam, self.rad, self.env

    def to_tensors(self):
        self.embeddings = torch.tensor(self.embeddings).float()
        self.gam = torch.tensor(self.gam).float()
        self.rad = torch.tensor(self.rad).float()
        self.env = torch.tensor(self.env).float()

    def resize_antenna(self):
        h, w = self.antenna_hw
        return cv2.resize(self.ant, (w, h))

    def load_antenna(self, antenna_folder):
        self.ant = np.load(os.path.join(antenna_folder, 'antenna.npy'))
        self.gam = downsample_gamma(np.load(os.path.join(antenna_folder, 'gamma.npy'))[np.newaxis], rate=4).squeeze()
        self.rad = downsample_radiation(np.load(os.path.join(antenna_folder, 'radiation.npy'))[np.newaxis],
                                        rates=[4, 2]).squeeze()
        self.env = np.load(os.path.join(antenna_folder, 'environment.npy'))
        if self.try_cache and os.path.exists(os.path.join(antenna_folder, 'embeddings.npy')):
            self.embeddings = np.load(os.path.join(antenna_folder, 'embeddings.npy'))



class AntennaDataSetsLoader:
    def __init__(self, dataset_path: str, batch_size: int, pca: PCA, split_ratio=None, try_cache=True):
        if split_ratio is None:
            split_ratio = [0.8, 0.19, 0.01]
        self.batch_size = batch_size
        self.split = split_ratio
        self.trn_folders, self.val_folders, self.tst_folders = [], [], []
        self.split_data(dataset_path, split_ratio)
        self.trn_dataset = AntennaDataSet(self.trn_folders, pca, try_cache)
        self.val_dataset = AntennaDataSet(self.val_folders, pca, try_cache)
        self.tst_dataset = AntennaDataSet(self.tst_folders, pca, try_cache)
        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, batch_size=batch_size)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, batch_size=batch_size)

    def split_data(self, dataset_path, split_ratio):
        all_folders = glob.glob(os.path.join(dataset_path, '[0-9]' * 5))
        random.seed(42)
        random.shuffle(all_folders)
        trn_len = int(len(all_folders) * split_ratio[0])
        val_len = int(len(all_folders) * split_ratio[1])
        tst_len = len(all_folders) - trn_len - val_len
        self.trn_folders = sorted(all_folders[:trn_len])
        self.val_folders = sorted(all_folders[trn_len:trn_len + val_len])
        self.tst_folders = sorted(all_folders[trn_len + val_len:])


def create_dataloader(gamma, radiation, params_scaled, batch_size, device, inv_or_forw='inverse'):
    gamma = torch.tensor(gamma).to(device).float()
    radiation = torch.tensor(radiation).to(device).float()
    params_scaled = torch.tensor(params_scaled).to(device).float()
    if inv_or_forw == 'inverse':
        dataset = torch.utils.data.TensorDataset(gamma, radiation, params_scaled)
    elif inv_or_forw == 'forward_gamma':
        dataset = torch.utils.data.TensorDataset(params_scaled, gamma)
    elif inv_or_forw == 'forward_radiation':
        dataset = torch.utils.data.TensorDataset(params_scaled, downsample_radiation(radiation, rates=[4, 2]))
    elif inv_or_forw == 'inverse_forward_gamma' or inv_or_forw == 'inverse_forward_GammaRad':
        dataset = torch.utils.data.TensorDataset(gamma, radiation)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


def nearest_neighbor_loss(loss_fn, x_train, y_train, x_val, y_val, k=1):
    strt = time.time()
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x_train)
    distances, indices = nbrs.kneighbors(x_val)
    cnt = len(np.where(distances < 0.1)[0])
    nearest_neighbor_y = y_train[indices].squeeze()
    loss = loss_fn(torch.tensor(nearest_neighbor_y), torch.tensor(y_val))
    return loss


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def split_dataset(dataset_path, train_val_test_split):
    folders = listdir(dataset_path)
    dataset_path_list = []
    folders = [folder for folder in folders if folder[:4] == 'wifi']
    for folder in folders:
        folder_path = join(dataset_path, folder)
        files = listdir(folder_path)
        for file in files:
            file_path = join(folder_path, file)
            if isfile(file_path):
                if file.endswith('.mat') and file.__contains__('results'):
                    dataset_path_list.append(file_path)

    dataset_path_list = np.array(dataset_path_list)
    num_of_data_points = len(dataset_path_list)
    num_of_train_points = int(num_of_data_points * train_val_test_split[0])
    num_of_val_points = int(num_of_data_points * train_val_test_split[1])

    train_pick = np.random.choice(num_of_data_points, num_of_train_points, replace=False)
    val_pick = np.random.choice(np.setdiff1d(np.arange(num_of_data_points), train_pick), num_of_val_points,
                                replace=False)
    if train_val_test_split[2] > 0:
        test_pick = np.setdiff1d(np.arange(num_of_data_points), np.concatenate((train_pick, val_pick)),
                                 assume_unique=True)
    else:
        test_pick = val_pick

    train_dataset_path_list = dataset_path_list[train_pick]
    val_dataset_path_list = dataset_path_list[val_pick]
    test_dataset_path_list = dataset_path_list[test_pick]
    return train_dataset_path_list, val_dataset_path_list, test_dataset_path_list


def create_dataset(dataset_path=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data',
                   train_val_test_split=[0.8, 0.1, 0.1]):
    dataset_path_list_train, dataset_path_list_val, dataset_path_list_test = split_dataset(dataset_path,
                                                                                           train_val_test_split)
    print('Creating dataset...')
    data_parameters_train, data_parameters_val, data_parameters_test = [], [], []
    data_gamma_train, data_gamma_val, data_gamma_test = [], [], []
    data_radiation_train, data_radiation_val, data_radiation_test = [], [], []

    for path in dataset_path_list_train:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters, np.array([0, 0, 19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
        rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
        data_radiation_train.append(rad_concat_swapped)
        data_parameters_train.append(parameters)
        data_gamma_train.append(gamma)

    for path in dataset_path_list_val:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters, np.array([0, 0, 19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
        rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
        data_radiation_val.append(rad_concat_swapped)
        data_parameters_val.append(parameters)
        data_gamma_val.append(gamma)

    for path in dataset_path_list_test:
        mat = sio.loadmat(path)
        parameters = np.squeeze(mat['parameters'])
        if path.__contains__('V1'):
            parameters = np.concatenate((parameters, np.array([0, 0, 19.55])))
        gamma = np.squeeze(mat['gamma'])
        gamma = np.concatenate((np.abs(gamma), np.angle(gamma)))
        rad = np.squeeze(mat['farfield'])[:, :, 1:, 0]
        rad_concat = np.concatenate((np.abs(rad), np.angle(rad)), axis=2)
        rad_concat_swapped = np.swapaxes(rad_concat, 0, 2)
        data_radiation_test.append(rad_concat_swapped)
        data_parameters_test.append(parameters)
        data_gamma_test.append(gamma)

    np.savez(dataset_path + '\\newdata.npz', parameters_train=np.array(data_parameters_train),
             gamma_train=np.array(data_gamma_train),
             radiation_train=np.array(data_radiation_train), parameters_val=np.array(data_parameters_val),
             gamma_val=np.array(data_gamma_val), radiation_val=np.array(data_radiation_val),
             parameters_test=np.array(data_parameters_test),
             gamma_test=np.array(data_gamma_test), radiation_test=np.array(data_radiation_test))
    print(f'Dataset created seccessfully. Saved in newdata.npz')


class standard_scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def forward(self, data):
        std_mask = self.std == 0
        std_copy = self.std.copy()
        std_copy[std_mask] = 1
        return (data - self.mean) / std_copy

    def inverse(self, data):
        return data * self.std + self.mean


class ScalerManager:
    def __init__(self, path: str):
        self.path = path
        self.scaler = None

    def try_loading_from_cache(self):
        if os.path.exists(self.path):
            cache_scaler = pickle.load(open(self.path, 'rb'))
            self.scaler = cache_scaler
            print(f'Cached scaler loaded from: {self.path}')

    def fit(self, data: Union[torch.tensor, np.ndarray]):
        self.scaler.fit(data)
        print(f'Scaler fitted')

    def dump(self):
        pickle.dump(self.scaler, open(self.path, 'wb'))
        print(f'Fitted scaler saved in: {self.path}')


class EnvironmentScalerLoader:
    def __init__(self, antenna_data_loader: torch.utils.data.DataLoader):
        self.antenna_data_loader = antenna_data_loader

    def load_environments(self):
        envs = []
        for i, X in enumerate(self.antenna_data_loader.trn_loader):
            _, _, _, env = X
            envs.append(env)
        envs = torch.cat(envs, dim=0).detach().cpu().numpy()
        return envs


def downsample_gamma(gamma, rate=4):
    gamma_len = gamma.shape[1]
    gamma_mag = gamma[:, :int(gamma_len / 2)]
    gamma_phase = gamma[:, int(gamma_len / 2):]
    gamma_mag_downsampled = gamma_mag[:, ::rate]
    gamma_phase_downsampled = gamma_phase[:, ::rate]
    gamma_downsampled = np.concatenate((gamma_mag_downsampled, gamma_phase_downsampled), axis=1)
    return gamma_downsampled


def downsample_radiation(radiation, rates=[4, 2]):
    first_dim_rate, second_dim_rate = rates
    radiation_downsampled = radiation[:, :, ::first_dim_rate, ::second_dim_rate]
    return radiation_downsampled


def convert_dataset_to_dB(data):
    print('Converting dataset to dB')
    train_params, train_gamma, train_radiation = data['parameters_train'], data['gamma_train'], data['radiation_train']
    val_params, val_gamma, val_radiation = data['parameters_val'], data['gamma_val'], data['radiation_val']
    test_params, test_gamma, test_radiation = data['parameters_test'], data['gamma_test'], data['radiation_test']
    train_gamma[:, :int(train_gamma.shape[1] / 2)] = 10 * np.log10(train_gamma[:, :int(train_gamma.shape[1] / 2)])
    val_gamma[:, :int(val_gamma.shape[1] / 2)] = 10 * np.log10(val_gamma[:, :int(val_gamma.shape[1] / 2)])
    test_gamma[:, :int(test_gamma.shape[1] / 2)] = 10 * np.log10(test_gamma[:, :int(test_gamma.shape[1] / 2)])
    train_radiation[:, :int(train_radiation.shape[1] / 2)] = 10 * np.log10(
        train_radiation[:, :int(train_radiation.shape[1] / 2)])
    val_radiation[:, :int(val_radiation.shape[1] / 2)] = 10 * np.log10(
        val_radiation[:, :int(val_radiation.shape[1] / 2)])
    test_radiation[:, :int(test_radiation.shape[1] / 2)] = 10 * np.log10(
        test_radiation[:, :int(test_radiation.shape[1] / 2)])
    np.savez(r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\newdata_dB.npz',
             parameters_train=train_params, gamma_train=train_gamma, radiation_train=train_radiation,
             parameters_val=val_params, gamma_val=val_gamma, radiation_val=val_radiation,
             parameters_test=test_params, gamma_test=test_gamma, radiation_test=test_radiation)
    print('Dataset converted to dB. Saved in data_dB.npz')


def produce_stats_gamma(GT_gamma, predicted_gamma, dataset_type='linear', to_print=True):
    if dataset_type == 'linear':
        GT_gamma[:, :int(GT_gamma.shape[1] / 2)] = 10 * np.log10(GT_gamma[:, :int(GT_gamma.shape[1] / 2)])
    if dataset_type == 'dB':
        pass
    if type(predicted_gamma) == tuple:
        GT_gamma, _ = GT_gamma
        predicted_gamma, _ = predicted_gamma
    predicted_gamma_mag, GT_gamma_mag = predicted_gamma[:, :int(predicted_gamma.shape[1] / 2)], GT_gamma[:, :int(
        GT_gamma.shape[1] / 2)]
    predicted_gamma_phase, GT_gamma_phase = predicted_gamma[:, int(predicted_gamma.shape[1] / 2):], GT_gamma[:, int(
        GT_gamma.shape[1] / 2):]
    # predicted_gamma_mag = 10*np.log10(predicted_gamma_mag)
    diff_dB = torch.abs(predicted_gamma_mag - GT_gamma_mag)
    respective_diff = torch.where(torch.abs(GT_gamma_mag) > 1.5, torch.div(diff_dB, torch.abs(GT_gamma_mag)) * 100, 0)
    avg_respective_diff = torch.mean(respective_diff[torch.nonzero(respective_diff, as_tuple=True)]).item()
    avg_diff = torch.mean(diff_dB, dim=1)
    max_diff = torch.max(diff_dB, dim=1)[0]
    avg_max_diff = torch.mean(max_diff).item()
    diff_phase = predicted_gamma_phase - GT_gamma_phase
    while len(torch.where(diff_phase > np.pi)[0]) > 0 or len(torch.where(diff_phase < -np.pi)[0]) > 0:
        diff_phase[torch.where(diff_phase > np.pi)] -= 2 * np.pi
        diff_phase[torch.where(diff_phase < -np.pi)] += 2 * np.pi
    diff_phase = torch.abs(diff_phase)
    avg_diff_phase = torch.mean(diff_phase, dim=1)
    max_diff_phase = torch.max(diff_phase, dim=1)[0]
    avg_max_diff_phase = torch.mean(max_diff_phase).item()
    if to_print:
        print('gamma- ' + dataset_type + ' dataset - Avg diff: {:.4f} dB, Avg dB respective diff: {:.4f} % ,'
                                         ' Avg max diff: {:.4f} dB, Avg diff phase: {:.4f} rad, Avg max diff phase: {:.4f} rad'
              .format(torch.mean(avg_diff).item(), avg_respective_diff, avg_max_diff, torch.mean(avg_diff_phase).item(),
                      avg_max_diff_phase))

    return avg_diff, max_diff, avg_diff_phase, max_diff_phase


def produce_radiation_stats(predicted_radiation, gt_radiation, to_print=True):
    if type(predicted_radiation) == tuple:
        _, predicted_radiation = predicted_radiation
        _, gt_radiation = gt_radiation
    sep = predicted_radiation.shape[1] // 2
    pred_rad_mag, gt_rad_mag = predicted_radiation[:, :sep], gt_radiation[:, :sep]
    pred_rad_phase, gt_rad_phase = predicted_radiation[:, sep:], gt_radiation[:, sep:]
    abs_diff_mag = torch.abs(pred_rad_mag - gt_rad_mag)
    respective_diff = torch.where(torch.abs(gt_rad_mag) > 1.5, torch.div(abs_diff_mag, torch.abs(gt_rad_mag)) * 100, 0)
    respective_diff = torch.mean(respective_diff[torch.nonzero(respective_diff, as_tuple=True)]).item()
    diff_phase = pred_rad_phase - gt_rad_phase
    while len(torch.where(diff_phase > np.pi)[0]) > 0 or len(torch.where(diff_phase < -np.pi)[0]) > 0:
        diff_phase[torch.where(diff_phase > np.pi)] -= 2 * np.pi
        diff_phase[torch.where(diff_phase < -np.pi)] += 2 * np.pi
    max_diff_mag = torch.amax(abs_diff_mag, dim=(1, 2, 3))
    mean_abs_error_mag = torch.mean(torch.abs(abs_diff_mag), dim=(1, 2, 3))
    mean_max_error_mag = torch.mean(max_diff_mag).item()
    abs_diff_phase = torch.abs(diff_phase)
    max_diff_phase = torch.amax(abs_diff_phase, dim=(1, 2, 3))
    mean_abs_error_phase = torch.mean(abs_diff_phase, dim=(1, 2, 3))
    mean_max_error_phase = torch.mean(max_diff_phase).item()
    msssim_vals = []
    for i in range(gt_radiation.shape[0]):
        msssim_vals.append(pytorch_msssim.msssim(pred_rad_mag[i:i + 1].float(), gt_rad_mag[i:i + 1].float()).item())
    msssim_vals = torch.tensor(msssim_vals)
    avg_msssim_mag = msssim_vals.mean().item()
    # print all the stats for prnt variant as one print statement
    if to_print:
        print('Radiation - mean_abs_error_mag:', round(torch.mean(mean_abs_error_mag).item(), 4),
              ' dB, mean dB respective error: ', round(respective_diff, 4)
              , '%, mean_max_error_mag:', round(mean_max_error_mag, 4)
              , ' dB, mean_abs_error_phase:', round(torch.mean(mean_abs_error_phase).item(), 4),
              ' rad, mean_max_error_phase:'
              , round(mean_max_error_phase, 4), ' rad, msssim_mag:', round(avg_msssim_mag, 4))
    return mean_abs_error_mag, max_diff_mag, msssim_vals


def save_antenna_mat(antenna: torch.Tensor, path: str, scaler: standard_scaler):
    import scipy.io as sio
    antenna = antenna.detach().cpu().numpy()
    antenna_unscaled = scaler.inverse(antenna)
    sio.savemat(path, {'antenna': antenna_unscaled})


class DXF2IMG(object):
    default_img_format = '.png'
    default_img_res = 30  # Adjust DPI for better quality

    def convert_dxf2img(self, names, img_format=default_img_format, img_res=default_img_res):
        for name in names:
            doc = ezdxf.readfile(name)
            msp = doc.modelspace()

            # Recommended: audit & repair DXF document before rendering
            auditor = doc.audit()
            if len(auditor.errors) != 0:
                raise Exception("The DXF document is damaged and can't be converted!")

            # Calculate figure size in inches to match desired pixel dimensions
            figsize_inches = (144 / img_res, 190 / img_res)
            fig = plt.figure(figsize=figsize_inches)
            ax = fig.add_axes([0, 0, 1, 1])
            ctx = RenderContext(doc)
            ctx.set_current_layout(msp)
            ctx.current_layout_properties.set_colors(bg='#FFFFFF')
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=True)

            # Construct the output image file path
            img_name = re.findall("(\S+)\.", name)[0]  # Select the image name that is the same as the DXF file name
            img_path = f"{img_name}{img_format}"
            fig.savefig(img_path, dpi=img_res)
            plt.close(fig)  # Close the figure to free up memory


def fill_lwpolylines_with_hatch(dxf_file_path, output_file_path=None, color=7):
    if output_file_path is None:
        output_file_path = dxf_file_path.replace('.dxf', '_hatch.dxf')
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()
    hatch_pattern = msp.add_hatch(color=color)
    for entity in msp:
        if entity.dxftype() == 'LWPOLYLINE':
            polyline = entity
            # Check if polyline is closed
            hatch_pattern.paths.add_polyline_path(polyline.vertices())

    doc.saveas(output_file_path)
    return output_file_path


def merge_dxf_files(input_files, output_file):
    # Create a new DXF document for the merged output
    merged_doc = ezdxf.new()

    for idx, file in enumerate(input_files):
        # Read each DXF file
        doc = ezdxf.readfile(file)
        msp = doc.modelspace()

        # Create a new layer for each input file
        layer_name = f'File_{idx + 1}'
        merged_doc.layers.new(name=layer_name)

        # Iterate through entities in the modelspace of the input file
        for entity in msp:
            # Copy each entity to the modelspace of the merged document
            merged_entity = entity.copy()  # Create a copy of the entity
            merged_entity.dxf.layer = layer_name  # Assign entity to the corresponding layer
            merged_doc.modelspace().add_entity(merged_entity)  # Add the copied entity to the merged modelspace

    # Save the merged DXF document to the output file
    merged_doc.saveas(output_file)


def run_dxf2img():
    dxf2img = DXF2IMG()
    all_images = []
    data_path = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs"
    for folder in sorted(os.listdir(data_path)):
        print('working on folder:', folder, end='\n')
        models_path = os.path.join(data_path, folder, 'models')
        for model_path in sorted(os.listdir(models_path)):
            merged_output_path = os.path.join(models_path, model_path, 'merged_hatch.dxf')
            if os.path.exists(merged_output_path):
                print('already processed:', model_path)
                continue
            print('working on model:', model_path)
            layer_path = os.path.join(models_path, model_path, 'layer_0_PEC.dxf')
            feed_pec_path = os.path.join(models_path, model_path, 'feed_PEC.dxf')
            feed_path = os.path.join(models_path, model_path, 'feed.dxf')
            layer_output_path = fill_lwpolylines_with_hatch(layer_path)
            feed_pec_output_path = fill_lwpolylines_with_hatch(feed_pec_path)
            feed_output_path = fill_lwpolylines_with_hatch(feed_path, color=1)
            merge_dxf_files([layer_output_path, feed_pec_output_path, feed_output_path], merged_output_path)
            dxf2img.convert_dxf2img([merged_output_path])

            pass


def gen2_gather_datasets(data_folder1, data_folder2, output_folder):
    output_file = os.path.join(output_folder, 'gen2_data.npz')
    if os.path.exists(output_file):
        print('Data already exists')
        return
    print('Gathering datasets')
    data_frequencies = np.load(os.path.join(data_folder1, 'frequencies.npy'))
    data_envs = np.concatenate((np.load(os.path.join(data_folder1, 'environments.npy')),
                                np.load(os.path.join(data_folder2, 'environments.npy'))), axis=0)
    data_gammas = np.concatenate((np.load(os.path.join(data_folder1, 'gammas.npy')).astype(np.float32),
                                  np.load(os.path.join(data_folder2, 'gammas.npy')).astype(np.float32)), axis=0)
    data_radiations = np.concatenate((np.load(os.path.join(data_folder1, 'radiations.npy')).astype(np.float32),
                                      np.load(os.path.join(data_folder2, 'radiations.npy')).astype(np.float32)), axis=0)
    np.savez(output_file, frequencies=data_frequencies, environments=data_envs, gammas=data_gammas,
             radiations=data_radiations)
    print('Data gathered successfully, saved in {}'.format(output_file))


def gen2_gather_antennas(data_folder1, data_folder2, output_folder):
    all_paths1 = [os.path.join(data_folder1, file, 'merged_hatch.png') for file in os.listdir(data_folder1)]
    all_paths2 = [os.path.join(data_folder2, file, 'merged_hatch.png') for file in os.listdir(data_folder2)]
    all_paths = all_paths1 + all_paths2
    for file in all_paths:
        antenna_img = cv2.imread(file)
        b, g, r = cv2.split(antenna_img)
        categorical_img = np.zeros_like(r, dtype=np.uint8)
        categorical_img[np.logical_and(np.logical_and(r == 255, b == 0), g == 0)] = 2  # feed is red
        categorical_img[np.mean(antenna_img, axis=2) > 200] = 1
        output_antenna_folder = os.path.join(output_folder, os.path.basename(Path(file).parent))
        if not os.path.exists(output_antenna_folder):
            os.makedirs(output_antenna_folder)
        output_antenna_file = os.path.join(output_antenna_folder, 'antenna.npy')
        np.save(output_antenna_file, categorical_img)
        cv2.imwrite(os.path.join(output_antenna_folder, 'antenna.png'), antenna_img)
        print('Antenna saved successfully in:', output_antenna_file)
    print('All antennas saved successfully')


def organize_dataset_per_antenna():
    antennas_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\gen2_antennas'
    other_data_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\gen2_data.npz'
    output_folder = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs'
    data = np.load(other_data_path)
    data_frequencies, data_envs, data_gammas, data_radiations = data['frequencies'], data['environments'], data[
        'gammas'], data['radiations']
    DataPreprocessor().assert_radiation_rules(data_radiations)
    DataPreprocessor().assert_gamma_rules(data_gammas)
    for idx in range(len(data_gammas)):
        antenna_path = os.path.join(output_folder, str(idx).zfill(5))
        print('Saving antenna number:', idx, 'to: ', antenna_path)
        if not os.path.exists(antenna_path):
            os.makedirs(antenna_path)
        shutil.copy(src=os.path.join(antennas_path, str(idx), 'antenna.npy'),
                    dst=os.path.join(antenna_path, 'antenna.npy'))
        shutil.copy(src=os.path.join(antennas_path, str(idx), 'antenna.png'),
                    dst=os.path.join(antenna_path, 'antenna.png'))
        np.save(os.path.join(antenna_path, 'gamma.npy'), data_gammas[idx])
        np.save(os.path.join(antenna_path, 'radiation.npy'), data_radiations[idx])
        np.save(os.path.join(antenna_path, 'environment.npy'), data_envs[idx])
        if idx == 0:
            np.save(os.path.join(antenna_path, 'frequencies.npy'), data_frequencies)
        print('Antenna number:', idx, 'saved successfully')
    pass


def set_plane_for_env(plane, start, stop):
    assert plane in ['xz'], 'Plane not supported yet'
    print('Setting plane for environment: ', plane)
    antennas_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs'
    for idx in range(start, stop):
        env_path = os.path.join(antennas_path, str(idx).zfill(5), 'environment.npy')
        env = np.load(env_path)
        env_with_plane = np.concatenate(([0], env))
        np.save(env_path, env_with_plane)
        print('Environment number:', idx, 'saved successfully')


if __name__ == '__main__':
    data_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs'
    ant_folders = [os.path.join(data_path, str(i).zfill(5)) for i in range(100)]
    pca = pickle.load(open(os.path.join(data_path, 'pca_model.pkl'), 'rb'))
    train_set = AntennaDataSet(ant_folders, pca)
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=10, shuffle=True)
    for i, data in enumerate(data_loader):
        print(data)
    # set_plane_for_env(plane='xz',start=0,stop=15000)
    # organize_dataset_per_antenna()
    # env_preprocessor = DataPreprocessor(folder_path=r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\data_2500x2\nits_checkpoints')
    # env_preprocessor.environment_preprocessor(debug=False)
    # data_processor = DataPreprocessor(folder_path=r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data"
    #                                               r"\data_15000_3envs\data_2500x2\results")
    # data_processor.gamma_preprocessor(debug=True)
    # data_processor.radiation_preprocessor(plot=False, debug=True)
    # data_path = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs\data_10000x1\processed_data"
    # gamma, radiation = np.load(os.path.join(data_path, 'gammas.npy')), np.load(
    #     os.path.join(data_path, 'radiations.npy'))
    # data_processor.assert_gamma_rules(gamma)
    # data_processor.assert_radiation_rules(radiation)
    # pass
    # run_dxf2img()
    # output_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_15000_3envs'
    # gen2_gather_antennas(data_folder1=os.path.join(output_path, 'data_10000x1', 'nits_checkpoints'),
    #                      data_folder2=os.path.join(output_path, 'data_2500x2', 'nits_checkpoints'),
    #                      output_folder=os.path.join(output_path, 'gen2_antennas'))
    #
    # gen2_gather_datasets(data_folder1=os.path.join(output_path, 'data_10000x1', 'processed_data'),
    #                      data_folder2=os.path.join(output_path, 'data_2500x2', 'processed_data'),
    #                      output_folder=output_path)
    # organize_dataset_per_antenna()
    pass
