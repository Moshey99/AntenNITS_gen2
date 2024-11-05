import copy
import shutil
from typing import Union, Optional
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
from AntennaDesign.pytorch_msssim import MSSSIM
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import time
import AntennaDesign.trainer
import AntennaDesign.losses
from AntennaDesign.models import baseline_regressor, inverse_hypernet
import random
import glob
import pickle
import re
import open3d as o3d
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from shapely.geometry import Polygon
from PCA_fitter.PCA_fitter_main import binarize


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
    def __init__(self, data_path=None, destination_path=None):
        if data_path is not None:
            self.num_data_points = len(os.listdir(data_path))
            self.folder_path = data_path
        if destination_path is not None:
            self.destination_path = destination_path
            os.makedirs(destination_path, exist_ok=True)

    def antenna_preprocessor(self, debug=False):
        scaler = standard_scaler()
        scaler_manager = ScalerManager(path=os.path.join(self.destination_path, 'ant_scaler.pkl'), scaler=scaler)
        print('Preprocessing antennas')
        all_ants = []
        folder_path = self.folder_path
        for i in sorted(os.listdir(folder_path)):
            print('working on antenna number:', i, 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, i, 'ant_parameters.pickle')
            with open(file_path, 'rb') as f:
                antenna_dict = pickle.load(f)
            antenna = np.array(list(antenna_dict.values()))
            all_ants.append(antenna)
            if not debug:
                output_folder = os.path.join(self.destination_path, i)
                os.makedirs(output_folder, exist_ok=True)
                np.save(os.path.join(output_folder, 'antenna.npy'), antenna)
        scaler_manager.fit(np.array(all_ants))
        scaler_manager.dump()
        print(f'Antennas saved successfully')

    def environment_preprocessor(self, debug=False):
        print('Preprocessing environments')
        scaler = standard_scaler()
        scaler_manager = ScalerManager(path=os.path.join(self.destination_path, 'env_scaler.pkl'), scaler=scaler)
        all_envs = []
        folder_path = self.folder_path
        for i in sorted(os.listdir(folder_path)):
            print('working on environment number:', i, 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, i, 'model_parameters.pickle')
            with open(file_path, 'rb') as f:
                env_dict = pickle.load(f)
                env_dict.pop('type', None)  # remove the first element which is the type of the mode (constant)
                plane = env_dict.get('plane', None)
                assert plane in ['xz', 'yz-flipped'], 'Plane parameter is missing in the environment'
                env_dict['plane'] = 0 if plane == 'xz' else 1
                env_vals = list(env_dict.values())
                assert np.all([type(value) != list for value in env_vals]), 'ERROR. List in Environments values'
                all_envs.append(env_vals)
                if not debug:
                    output_folder = os.path.join(self.destination_path, i)
                    os.makedirs(output_folder, exist_ok=True)
                    np.save(os.path.join(output_folder, 'environment.npy'), np.array(env_vals))
        scaler_manager.fit(np.array(all_envs))
        scaler_manager.dump()
        print(f'Environments saved successfully')

    def radiation_preprocessor(self, selected_frequencies=None, plot=False, debug=False):
        print('Preprocessing radiations')
        if selected_frequencies is None:
            selected_frequencies = [1500, 2100, 2400]
        folder_path = self.folder_path
        for idx, i in enumerate(os.listdir(folder_path)):
            print('working on antenna number:', i, 'out of:', self.num_data_points)
            sample_folder = os.path.join(folder_path, i)
            efficiency = self.load_efficiency(sample_folder)
            all_frequencies = []
            for freq in selected_frequencies:
                freq_efficiency = self.find_efficiency_for_f(efficiency, freq)
                file_path = os.path.join(sample_folder, f'farfield (f={freq}) [1].txt')
                im_resized = np.zeros((4, 181, 91))
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
                    current_im = np.clip(cv2.resize(im_resh[j], (91, 181), interpolation=cv2.INTER_LINEAR),
                                         im_resh[j].min(), im_resh[j].max())
                    if j < 2:
                        assert np.all(current_im >= 0), 'Negative values in radiation magnitude'
                        current_im = current_im * freq_efficiency
                        current_im = 10 * np.log10(current_im)
                    else:
                        assert np.all(current_im >= 0) and np.all(current_im <= 360), 'Phase values out of range 0-360'
                        current_im = np.deg2rad(current_im) - np.pi

                    im_resized[j] = current_im
                    if plot:
                        titles = ['mag1', 'mag2', 'phase1', 'phase2']
                        plt.subplot(2, 2, j + 1)
                        plt.imshow(current_im)
                        plt.title(titles[j])
                        plt.colorbar()
                        plt.show() if j == im_resh.shape[0] - 1 else None
                all_frequencies.append(im_resized)

            radiation_multi_frequencies = np.array(all_frequencies)
            mag_db, phase_radians = radiation_multi_frequencies[:, :2], radiation_multi_frequencies[:, 2:]
            mag_db_concat = mag_db.reshape(-1, mag_db.shape[2], mag_db.shape[3])
            phase_radians_concat = phase_radians.reshape(-1, phase_radians.shape[2], phase_radians.shape[3])
            radiations = np.concatenate((mag_db_concat, phase_radians_concat), axis=0).astype(np.float32)
            self.assert_radiation_rules(radiations[np.newaxis])
            if not debug:
                saving_folder = os.path.join(self.destination_path, i)
                os.makedirs(saving_folder, exist_ok=True)
                np.save(os.path.join(saving_folder, 'radiation.npy'), radiations)

    def gamma_preprocessor(self, debug=False):
        print('Preprocessing gammas')
        folder_path = self.folder_path
        for i in sorted(os.listdir(folder_path)):
            print('working on antenna number:', i, 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, i, 'S_parameters.pickle')
            with open(file_path, 'rb') as f:
                gamma_raw = pickle.load(f)
                gamma_complex = gamma_raw[0]
                gamma_mag, gamma_phase = np.abs(gamma_complex), np.angle(
                    gamma_complex)  # gamma_phase in radians already
                gamma_mag_dB = 10 * np.log10(gamma_mag)
                gamma = np.concatenate((gamma_mag_dB, gamma_phase))
                self.assert_gamma_rules(gamma[np.newaxis])
                if not debug:
                    output_folder = os.path.join(self.destination_path, i)
                    os.makedirs(output_folder, exist_ok=True)
                    np.save(os.path.join(output_folder, 'gamma.npy'), gamma)
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

    @staticmethod
    def assert_gamma_rules(gamma: np.ndarray):
        eps = 1e-6
        mag_db = gamma[:, :int(gamma.shape[1] / 2)]
        phase_rad = gamma[:, int(gamma.shape[1] / 2):]
        mag_linear = 10 ** (mag_db / 10)
        assert np.all(mag_linear >= 0), 'Negative values in gamma magnitude'
        assert np.all(phase_rad >= -np.pi - eps) and np.all(
            phase_rad <= np.pi + eps), 'Phase values out of range -pi - pi radians'

    @staticmethod
    def load_efficiency(folder):
        efficiency_file = os.path.join(folder, 'Efficiency.pickle')
        with open(efficiency_file, 'rb') as f:
            efficiency = np.real(pickle.load(f))
            return efficiency

    @staticmethod
    def find_efficiency_for_f(efficiency, freq):
        freqs = efficiency[2, :]
        idx = np.where(freqs == freq)[0]
        return efficiency[0, idx]


class PCAWrapper:
    def __init__(self, pca: Optional[PCA]):
        self.pca = pca

    @property
    def pca_components_std(self) -> np.ndarray:
        return np.sqrt(self.pca.explained_variance_)

    def normalize_principal_components(self, components: np.ndarray) -> np.ndarray:
        return components / (self.pca_components_std + 1e-7)

    def unnormalize_principal_components(self, components: np.ndarray) -> np.ndarray:
        return components * (self.pca_components_std + 1e-7)

    def image_from_components(self, components: np.ndarray, shape=(144, 200)):
        n_samples, n_features = components.shape
        ant_resized = self.pca.inverse_transform(components).reshape((n_samples, *shape))
        return ant_resized

    def components_from_image(self, image: np.ndarray) -> np.ndarray:
        n_samples, h, w = image.shape
        return self.pca.transform(image.flatten().reshape(n_samples, -1))

    def apply_binarization_on_components(self, components: np.ndarray) -> np.ndarray:
        image = self.image_from_components(components)
        image_binarized = binarize(image)
        return self.components_from_image(image_binarized)

    def apply_unnormalization_and_binarization_on_components(self, components: np.ndarray) -> np.ndarray:
        components_unnormalized = self.unnormalize_principal_components(components)
        return self.apply_binarization_on_components(components_unnormalized)


class AntennaDataSet(torch.utils.data.Dataset):
    def __init__(self, antenna_folders: list[str], pca_wrapper: PCAWrapper, try_cache: bool):
        self.antenna_folders = antenna_folders
        self.len = len(antenna_folders)
        self.pca_wrapper = pca_wrapper
        self.try_cache = try_cache
        self.antenna_hw = (144, 200)  # self.antenna_hw = (20, 20)
        self.ant, self.embeddings, self.gam, self.rad, self.env = None, None, None, None, None
        self.shapes = None
        self.get_shapes()

    def get_shapes(self):
        self.load_antenna(self.antenna_folders[0])
        self.shapes = {'ant': self.ant.shape, 'gam': self.gam.shape, 'rad': self.rad.shape, 'env': self.env.shape}
        self.__reset_all()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        antenna_folder = self.antenna_folders[idx]
        antenna_name = os.path.basename(antenna_folder)
        self.load_antenna(antenna_folder)
        self.get_embeddings()
        self.to_tensors()
        # np.random.seed(42+idx)
        # embs = np.random.rand()*np.ones(10)
        # embs = torch.tensor(embs).float()
        embs = self.embeddings.detach().clone()
        self.embeddings = None
        antenna_full_name = [f'SPEC_{antenna_name}_ENV_{env_name}' for env_name in self.env_names]
        return embs, self.gam, self.rad, self.env, antenna_full_name

    def get_embeddings(self):
        if self.pca_wrapper.pca is not None:
            ant_resized = self.resize_antenna()
            if self.embeddings is None:
                self.embeddings = self.pca_wrapper.pca.transform(ant_resized.flatten().reshape(1, -1)).flatten()
            else:
                self.embeddings = copy.deepcopy(ant_resized)
        else:
            self.embeddings = copy.deepcopy(self.ant)

    def to_tensors(self):
        self.embeddings = torch.tensor(self.embeddings).float()
        self.gam = torch.tensor(self.gam).float()
        self.rad = torch.tensor(self.rad).float()
        self.env = torch.tensor(self.env).float()

    def resize_antenna(self):
        h, w = self.antenna_hw
        return cv2.resize(self.ant, (w, h))

    def load_antenna(self, antenna_folder):
        self.ant = np.zeros(40) #np.load(os.path.join(antenna_folder, 'antenna.npy'))
        self.gam = downsample_gamma(np.load(os.path.join(antenna_folder, 'gamma.npy'))[np.newaxis], rate=4).squeeze()
        self.rad = downsample_radiation(np.load(os.path.join(antenna_folder, 'radiation.npy'))[np.newaxis],
                                        rates=[4, 2]).squeeze()
        self.rad = self.__clip_radiation(self.rad)
        environment_folders = os.path.join(antenna_folder, '..', '..', 'environments')
        self.env_names = sorted(os.listdir(environment_folders))
        self.env = np.array([np.load(os.path.join(environment_folders, env_name, 'environment.npy'))
                    for env_name in self.env_names])
        if self.try_cache and os.path.exists(os.path.join(antenna_folder, 'embeddings.npy')):
            self.embeddings = np.load(os.path.join(antenna_folder, 'embeddings.npy'))

    @staticmethod
    def __clip_radiation(radiation: np.ndarray):
        assert len(radiation.shape) == 3, 'Radiation shape is not 3D (channels, h, w)'
        radiation[:int(radiation.shape[0] / 2)] = np.clip(radiation[:int(radiation.shape[0] / 2)], -20, 5)
        return radiation

    def __reset_all(self):
        self.ant, self.embeddings, self.gam, self.rad, self.env = None, None, None, None, None


class AntennaDataSetsLoader:
    def __init__(self, dataset_path: str, batch_size: int, pca: Optional[PCA] = None, split_ratio=None, try_cache=True):
        if split_ratio is None:
            split_ratio = [1, 0, 0]
        self.pca_wrapper = PCAWrapper(pca)
        self.batch_size = batch_size
        self.split = split_ratio
        self.trn_folders, self.val_folders, self.tst_folders = [], [], []
        self.split_data(dataset_path, split_ratio)
        self.trn_dataset = AntennaDataSet(self.trn_folders, self.pca_wrapper, try_cache)
        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, batch_size=batch_size)
        # self.val_dataset = AntennaDataSet(self.val_folders, self.pca_wrapper, try_cache)
        # self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size)
        # self.tst_dataset = AntennaDataSet(self.tst_folders, self.pca_wrapper, try_cache)
        # self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, batch_size=batch_size)

    def split_data(self, dataset_path, split_ratio):
        # all_folders = sorted(glob.glob(os.path.join(dataset_path, '[0-9]*')))
        # all_folders = [folder for folder in all_folders if
        #                os.path.basename(folder).startswith('13') or os.path.basename(folder).startswith('14')]
        all_folders = [os.path.join(dataset_path, path) for path in sorted(os.listdir(dataset_path))]
        all_folders = [folder for folder in all_folders if not os.path.isfile(folder) and 'checkpoints' not in folder]
        random.seed(42)
        random.shuffle(all_folders)
        trn_len = int(len(all_folders) * split_ratio[0])
        val_len = int(len(all_folders) * split_ratio[1])
        tst_len = len(all_folders) - trn_len - val_len
        self.trn_folders = all_folders[:trn_len]
        self.val_folders = all_folders[trn_len:trn_len + val_len]
        self.tst_folders = all_folders[trn_len + val_len:]


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
        assert self.mean is not None and self.std is not None, 'Scaler is not fitted'
        std_mask = self.std < 1e-7
        std_copy = self.std.copy()
        std_copy[std_mask] = 1
        return (data - self.mean) / std_copy

    def inverse(self, data):
        assert self.mean is not None and self.std is not None, 'Scaler is not fitted'
        return data * self.std + self.mean


class ScalerManager:
    def __init__(self, path: str, scaler: Optional[standard_scaler] = None):
        self.path = path
        self.scaler = scaler

    def try_loading_from_cache(self):
        if os.path.exists(self.path):
            cache_scaler = pickle.load(open(self.path, 'rb'))
            self.scaler = cache_scaler
            print(f'Cached scaler loaded from: {self.path}')

    def fit(self, data: np.ndarray):
        assert self.scaler is not None, 'Scaler is not initialized'
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
            print(f'Loading environment {i} out of {len(self.antenna_data_loader.trn_loader)}')
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


def gamma_to_dB(gamma: torch.Tensor):
    gamma_dB = gamma.clone()
    gamma_dB[:, :int(gamma.shape[1] / 2)] = 10 * torch.log10(gamma[:, :int(gamma.shape[1] / 2)])
    return gamma_dB


def gamma_mag_to_dB(gamma_mag: torch.Tensor):
    gamma_mag_dB = 10 * torch.log10(gamma_mag)
    return gamma_mag_dB


def gamma_to_linear(gamma: torch.Tensor):
    gamma_linear = gamma.clone()
    gamma_linear[:, :int(gamma.shape[1] / 2)] = 10 ** (gamma[:, :int(gamma.shape[1] / 2)] / 10)
    return gamma_linear


def gamma_mag_to_linear(gamma_mag: torch.Tensor):
    gamma_mag_linear = 10 ** (gamma_mag / 10)
    return gamma_mag_linear


def radiation_to_dB(radiation: torch.Tensor):
    radiation_dB = radiation.clone()
    radiation_dB[:, :int(radiation.shape[1] / 2)] = 10 * torch.log10(radiation[:, :int(radiation.shape[1] / 2)])
    return radiation_dB


def radiation_mag_to_dB(radiation_mag: torch.Tensor):
    radiation_mag_dB = 10 * torch.log10(radiation_mag)
    return radiation_mag_dB


def radiation_to_linear(radiation: torch.Tensor):
    radiation_linear = radiation.clone()
    radiation_linear[:, :int(radiation.shape[1] / 2)] = 10 ** (radiation[:, :int(radiation.shape[1] / 2)] / 10)
    return radiation_linear


def radiation_mag_to_linear(radiation_mag: torch.Tensor):
    radiation_mag_linear = 10 ** (radiation_mag / 10)
    return radiation_mag_linear


def produce_gamma_stats(GT_gamma, predicted_gamma, dataset_type='linear', to_print=True):
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
    respective_diff = torch.where(GT_gamma_mag < -1.5, torch.div(diff_dB, torch.abs(GT_gamma_mag)) * 100, 0)
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
    msssim = MSSSIM()
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
        msssim_vals.append(msssim(pred_rad_mag.float(), gt_rad_mag[i:i + 1].float()).item())
    msssim_vals = torch.tensor(msssim_vals)
    avg_msssim_mag = msssim_vals.mean().item()
    if np.isnan(avg_msssim_mag):
        pass
    # print all the stats for prnt variant as one print statement
    if to_print:
        print('Radiation - mean_abs_error_mag:', round(torch.mean(mean_abs_error_mag).item(), 4),
              ' dB, mean dB respective error: ', round(respective_diff, 4)
              , '%, mean_max_error_mag:', round(mean_max_error_mag, 4)
              , ' dB, mean_abs_error_phase:', round(torch.mean(mean_abs_error_phase).item(), 4),
              ' rad, mean_max_error_phase:'
              , round(mean_max_error_phase, 4), ' rad, msssim_mag:', round(avg_msssim_mag, 4))
    return mean_abs_error_mag, max_diff_mag, mean_abs_error_phase, msssim_vals


def save_antenna_mat(antenna: torch.Tensor, path: str, scaler: standard_scaler):
    import scipy.io as sio
    antenna = antenna.detach().cpu().numpy()
    antenna_unscaled = scaler.inverse(antenna)
    sio.savemat(path, {'antenna': antenna_unscaled})


class DXF2IMG(object):
    default_img_format = '.png'
    default_img_res = 12  # Adjust DPI for better quality

    def convert_dxf2img(self, names, img_format=default_img_format, img_res=default_img_res):
        def fig2img(fig):
            from PIL import Image
            """Convert a Matplotlib figure to a PIL Image and return it"""
            import io
            buf = io.BytesIO()
            fig.savefig(buf)
            buf.seek(0)
            img = Image.open(buf)
            return img

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
            p = ax.transData.transform((19, 26))
            img = np.array(fig2img(fig))
            plt.figure()
            plt.imshow(img)

            # Construct the output image file path
            plt.show()
            img_name = re.findall("(\S+)\.", name)[0]  # Select the image name that is the same as the DXF file name
            img_path = f"{img_name}{img_format}"
            fig.savefig(img_path)
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
    data_path = r"C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\simplified_dataset"
    for model_path in sorted(os.listdir(data_path)):
        merged_output_path = os.path.join(data_path, model_path, 'merged_hatch.dxf')
        # if os.path.exists(merged_output_path):
        #     print('already processed:', model_path)
        #     # continue
        print('working on model:', model_path)
        layer_path = os.path.join(data_path, model_path, 'layer_0_PEC.dxf')
        feed_pec_path = os.path.join(data_path, model_path, 'feed_PEC.dxf')
        feed_path = os.path.join(data_path, model_path, 'feed.dxf')
        layer_output_path = fill_lwpolylines_with_hatch(layer_path)
        feed_pec_output_path = fill_lwpolylines_with_hatch(feed_pec_path, color=2)
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


def gen2_gather_antennas(output_folder, data_folder1, data_folder2=None):
    all_paths1 = [os.path.join(data_folder1, file, 'merged_hatch.png') for file in os.listdir(data_folder1)]
    if data_folder2 is not None:
        all_paths2 = [os.path.join(data_folder2, file, 'merged_hatch.png') for file in os.listdir(data_folder2)]
        all_paths = all_paths1 + all_paths2
    else:
        all_paths = all_paths1
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


def gen2_organize_from_npz(npz_file, output_folder):
    data = np.load(npz_file)
    radiations, gammas, idxs = data['radiation'], data['gamma'], data['index']
    for i, idx in enumerate(idxs):
        antenna_path = os.path.join(output_folder, str(idx).zfill(5))
        assert os.path.exists(antenna_path)
        np.save(os.path.join(antenna_path, 'gamma.npy'), gammas[i])
        np.save(os.path.join(antenna_path, 'radiation.npy'), radiations[i])
        print('gamma and radiation for number:', idx, 'saved successfully')
        pass


def get_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is a symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def check_ant_validity(ant_parameters, model_parameters):
    Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters['w'] / 2
          - model_parameters['feed_length'] / 2)
    Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
    wings = ['w1', 'w2', 'q1', 'q2']
    for wing in wings:
        if (ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z2'] and
                ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z2'] > ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z3'] and
                ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z2'] and
                ant_parameters[f'{wing}y3'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z2'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z1'] and
                ant_parameters[f'{wing}y3'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z2'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z1'] and
                ant_parameters[f'{wing}y2'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y3']):
            return 0
        if (ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z2'] and
                ant_parameters[f'{wing}y2'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y3']):
            return 0
        if np.abs(ant_parameters[f'{wing}z2'] - ant_parameters[f'{wing}z1']) < ant_parameters['w'] / Sz: return 0
        if np.abs(ant_parameters[f'{wing}z1'] - ant_parameters[f'{wing}z3']) < ant_parameters['w'] / Sz: return 0
        if np.abs(ant_parameters[f'{wing}z2'] - ant_parameters[f'{wing}z3']) < ant_parameters['w'] / Sz: return 0
        if ant_parameters[f'{wing}y1'] < ant_parameters['w'] / Sy: return 0
        if np.abs(ant_parameters[f'{wing}y2'] - ant_parameters[f'{wing}y1']) < ant_parameters['w'] / Sy: return 0
        if np.abs(ant_parameters[f'{wing}y1'] - ant_parameters[f'{wing}y3']) < ant_parameters['w'] / Sy: return 0
        if np.abs(ant_parameters[f'{wing}y2'] - ant_parameters[f'{wing}y3']) < ant_parameters['w'] / Sy: return 0
    if (Sz * ant_parameters[f'q1z3'] - ant_parameters['w'] / 2 <= 5
            and (ant_parameters[f'q1y3'] < ant_parameters['fx'] < ant_parameters[f'q1y2'] or
                 ant_parameters[f'q1y2'] < ant_parameters['fx'] < ant_parameters[f'q1y3'])): return 0
    if (Sz * ant_parameters[f'w1z3'] - ant_parameters['w'] / 2 <= 5
            and (ant_parameters[f'w1y3'] < ant_parameters['fx'] < ant_parameters[f'w1y2'] or
                 ant_parameters[f'w1y2'] < ant_parameters['fx'] < ant_parameters[f'w1y3'])): return 0
    wings = ['w1', 'w2', 'w3', 'q1', 'q2', 'q3']
    for wing in wings:
        if np.abs(ant_parameters[f'{wing}z0'] - ant_parameters[f'{wing}z1']) < ant_parameters['w'] / Sz: return 0
    if np.min([ant_parameters[f'q3z0'], ant_parameters[f'w3z0']]) > 0.2: return 0
    return 1


class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72. / self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([], [], **kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw = ((self.trans((1, self.lw_data)) - self.trans((0, 0))) * self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda: self.fig.canvas.draw_idle())
        self.timer.start()


def plot_antenna_figure(model_parameters, ant_parameters, alpha=1):
    plt.ioff()
    f, ax1 = plt.subplots()
    wings = ['w1', 'w2', 'q1', 'q2']
    Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters['w'] / 2
          - model_parameters['feed_length'] / 2)
    Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
    data_linewidth_plot([Sy * ant_parameters['fx'], Sy * ant_parameters['fx']],
                        [-10, 10], linewidth=ant_parameters['w'] + 0.1, alpha=alpha, color='k')
    for wing in wings:
        if wing[0] == 'q':
            sign = -1
        else:
            sign = 1
        z = [Sz * ant_parameters[f'{wing}z0']]
        y = [0, 0]
        for i1 in range(3):
            z.append(Sz * ant_parameters[f'{wing}z{i1 + 1:d}'])
            z.append(Sz * ant_parameters[f'{wing}z{i1 + 1:d}'])
            y.append(Sy * ant_parameters[f'{wing}y{i1 + 1:d}'])
            y.append(Sy * ant_parameters[f'{wing}y{i1 + 1:d}'])
        y.pop()
        data_linewidth_plot(y, sign * np.array(z),
                            linewidth=ant_parameters['w'], alpha=alpha, color='b')
    wings = ['w3', 'q3']
    for wing in wings:
        if wing[0] == 'q':
            sign = -1
        else:
            sign = 1
        z = [Sz * ant_parameters[f'{wing}z0']]
        y = [Sy * ant_parameters['fx'], Sy * ant_parameters['fx']]
        z.append(Sz * ant_parameters[f'{wing}z{1:d}'])
        z.append(Sz * ant_parameters[f'{wing}z{1:d}'])
        y.append(Sy * ant_parameters[f'{wing}y{1:d}'])
        data_linewidth_plot(y, sign * np.array(z),
                            linewidth=ant_parameters['w'], alpha=alpha, color='b')
    data_linewidth_plot([0, 0],
                        [model_parameters['feed_length'] / 2, -model_parameters['feed_length'] / 2],
                        linewidth=ant_parameters['w'], alpha=alpha, color='w')
    data_linewidth_plot([Sy * ant_parameters['fx'], Sy * ant_parameters['fx']],
                        [model_parameters['feed_length'] / 2, -model_parameters['feed_length'] / 2],
                        linewidth=ant_parameters['w'] + 0.1, alpha=alpha, color='r')
    plt.title('dimensions in mm')
    ax1.set_aspect('equal')
    # plt.show()
    return f


if __name__ == '__main__':
    data_path = r'C:\Users\moshey\PycharmProjects\etof_folder_git\AntennaDesign_data\data_110k_150k_raw'
    destination_path = data_path.replace(os.path.basename(data_path), 'data_110k_150k_processed')
    preprocessor = DataPreprocessor(data_path=data_path, destination_path=destination_path)
    preprocessor.antenna_preprocessor()
    preprocessor.environment_preprocessor()
    preprocessor.radiation_preprocessor()
    preprocessor.gamma_preprocessor()
    pass
