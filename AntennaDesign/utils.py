import copy
import shutil
from typing import Union, Optional, List, Tuple
import cv2
import numpy
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

MODEL_TYPE = 5
assert MODEL_TYPE in [3, 5], 'MODEL_TYPE must be either 3 or 5.'
EXAMPLE_FOLDER = os.path.join(Path(__file__).parent, 'EXAMPLE', f'model_{MODEL_TYPE}')


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
        ant_path = os.path.join(EXAMPLE_FOLDER, 'ant_parameters.pickle')
        with open(ant_path, 'rb') as f:
            example = pickle.load(f)
        print('Preprocessing antennas')
        all_ants = []
        folder_path = self.folder_path
        for idx, name in enumerate(sorted(os.listdir(folder_path))):
            print('working on antenna number:', name, f'({idx})', 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, name, 'ant_parameters.pickle')
            with open(file_path, 'rb') as f:
                antenna_dict = pickle.load(f)
            antenna_vals = [antenna_dict[key] for key in example.keys()]
            antenna = np.array(antenna_vals)
            all_ants.append(antenna)
            if not debug:
                output_folder = os.path.join(self.destination_path, name)
                os.makedirs(output_folder, exist_ok=True)
                np.save(os.path.join(output_folder, 'antenna.npy'), antenna)
        scaler_manager.fit(np.array(all_ants))
        scaler_manager.dump()
        print(f'Antennas saved successfully')

    def environment_preprocessor(self, debug=False):
        print('Preprocessing environments')
        scaler = standard_scaler()
        scaler_manager = ScalerManager(path=os.path.join(self.destination_path, 'env_scaler.pkl'), scaler=scaler)
        env_path = os.path.join(EXAMPLE_FOLDER, 'model_parameters.pickle')
        with open(env_path, 'rb') as f:
            example = pickle.load(f)
        all_envs = []
        folder_path = self.folder_path
        for idx, name in enumerate(sorted(os.listdir(folder_path))):
            print('working on antenna number:', name, f'({idx})', 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, name, 'model_parameters.pickle')
            with open(file_path, 'rb') as f:
                env_dict = pickle.load(f)
                env_dict.pop('type', None)  # remove the first element which is the type of the mode (constant)
                example.pop('type', None)
                plane = env_dict.get('plane', None)
                assert plane in ['xz', 'yz-flipped'], 'Plane parameter is missing in the environment'
                env_dict['plane'] = 0 if plane == 'xz' else 1
                env_vals = [env_dict[key] for key in example.keys()]
                assert np.all([type(value) != list for value in env_vals]), 'ERROR. List in Environments values'
                all_envs.append(env_vals)
                if not debug:
                    output_folder = os.path.join(self.destination_path, name)
                    os.makedirs(output_folder, exist_ok=True)
                    np.save(os.path.join(output_folder, 'environment.npy'), np.array(env_vals))
        scaler_manager.fit(np.array(all_envs))
        scaler_manager.dump()
        print(f'Environments saved successfully')

    @staticmethod
    def default_selected_frequencies():
        return [1500, 2100, 2400]

    def radiation_preprocessor(self, mode: str = 'directivity', selected_frequencies=None, plot=False, debug=False):
        assert mode in ['directivity', 'gain'], 'Invalid radiation mode'
        print('Preprocessing radiations')
        if selected_frequencies is None:
            selected_frequencies = self.default_selected_frequencies()
        folder_path = self.folder_path
        for idx, name in enumerate(sorted(os.listdir(folder_path))):
            print('working on antenna number:', name, f'({idx})', 'out of:', self.num_data_points)
            sample_folder = os.path.join(folder_path, name)
            efficiency = self.load_efficiency(sample_folder)
            if efficiency is None:
                print(f'Efficiency file is missing for antenna {name}. Skipping this antenna.')
                continue
            all_frequencies = []
            for freq in selected_frequencies:
                freq_efficiency = 1 if mode == 'directivity' else self.find_efficiency_for_f(efficiency, freq)
                file_path = os.path.join(sample_folder, f'farfield (f={freq}) [1].txt')
                radiation_resized = np.zeros((4, 181, 91))
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
                    current_ch = np.clip(cv2.resize(im_resh[j], (91, 181), interpolation=cv2.INTER_LINEAR),
                                         im_resh[j].min(), im_resh[j].max())
                    if j < 2:
                        assert np.all(current_ch >= 0), 'Negative values in radiation magnitude'
                        current_ch = current_ch * freq_efficiency
                        current_ch = 10 * np.log10(current_ch + 1e-7)
                    else:
                        assert np.all(current_ch >= 0) and np.all(current_ch <= 360), 'Phase values out of range 0-360'
                        current_ch = np.deg2rad(current_ch) - np.pi

                    radiation_resized[j] = current_ch
                    if plot:
                        titles = ['mag1', 'mag2', 'phase1', 'phase2']
                        plt.subplot(2, 2, j + 1)
                        plt.imshow(current_ch)
                        plt.title(titles[j])
                        plt.colorbar()
                        plt.show() if j == im_resh.shape[0] - 1 else None
                all_frequencies.append(radiation_resized)

            radiation_multi_frequencies = np.array(all_frequencies)
            mag_db, phase_radians = radiation_multi_frequencies[:, :2], radiation_multi_frequencies[:, 2:]
            mag_db_concat = mag_db.reshape(-1, mag_db.shape[2], mag_db.shape[3])
            phase_radians_concat = phase_radians.reshape(-1, phase_radians.shape[2], phase_radians.shape[3])
            radiations = np.concatenate((mag_db_concat, phase_radians_concat), axis=0).astype(np.float32)
            self.assert_radiation_rules(radiations[np.newaxis])
            if not debug:
                saving_folder = os.path.join(self.destination_path, name)
                os.makedirs(saving_folder, exist_ok=True)
                file_name = 'radiation_directivity' if mode == 'directivity' else 'radiation'
                file_path = os.path.join(saving_folder, file_name + '.npy')
                np.save(file_path, radiations)

    def gamma_preprocessor(self, debug=False):
        print('Preprocessing gammas')
        folder_path = self.folder_path
        for idx, name in enumerate(sorted(os.listdir(folder_path))):
            print('working on antenna number:', name, f'({idx})', 'out of:', self.num_data_points)
            file_path = os.path.join(folder_path, name, 'S_parameters.pickle')
            if not os.path.exists(file_path):
                print(f'Gamma file is missing for antenna {name}. Skipping this antenna.')
                continue
            with open(file_path, 'rb') as f:
                gamma_raw = pickle.load(f)
                gamma_complex = gamma_raw[0]
                gamma_mag, gamma_phase = np.abs(gamma_complex), np.angle(
                    gamma_complex)  # gamma_phase in radians already
                gamma_mag_dB = 20 * np.log10(gamma_mag)
                gamma = np.concatenate((gamma_mag_dB, gamma_phase))
                self.assert_gamma_rules(gamma[np.newaxis])
                if not debug:
                    output_folder = os.path.join(self.destination_path, name)
                    os.makedirs(output_folder, exist_ok=True)
                    np.save(os.path.join(output_folder, 'gamma.npy'), gamma)
        print('Gammas saved successfully with mag in dB and phase in radians')
        pass

    @staticmethod
    def assert_radiation_rules(radiation: np.ndarray):
        eps = 1e-6
        mag_db = radiation[:, :int(radiation.shape[1] / 2)]
        phase_rad = radiation[:, int(radiation.shape[1] / 2):]
        mag_linear = radiation_mag_to_linear(mag_db)
        assert np.all(mag_linear >= 0), 'Negative values in radiation magnitude'
        assert np.all(phase_rad >= -np.pi - eps) and np.all(
            phase_rad <= np.pi + eps), 'Phase values out of range -pi - pi radians'

    @staticmethod
    def assert_gamma_rules(gamma: np.ndarray):
        eps = 1e-6
        mag_db = gamma[:, :int(gamma.shape[1] / 2)]
        phase_rad = gamma[:, int(gamma.shape[1] / 2):]
        mag_linear = gamma_mag_to_linear(mag_db)
        assert np.all(mag_linear >= 0), 'Negative values in gamma magnitude'
        assert np.all(phase_rad >= -np.pi - eps) and np.all(
            phase_rad <= np.pi + eps), 'Phase values out of range -pi - pi radians'

    @staticmethod
    def load_efficiency(folder):
        efficiency_file = os.path.join(folder, 'Efficiency.pickle')
        if os.path.exists(efficiency_file):
            with open(efficiency_file, 'rb') as f:
                efficiency = np.real(pickle.load(f))
                return efficiency
        else:
            return None

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
        image_binarized = self.binarize(image)
        return self.components_from_image(image_binarized)

    @staticmethod
    def binarize(img, nonmetal_threshold=0.5, feed_threshold=1.5):
        img[img < nonmetal_threshold] = 0
        img[img >= feed_threshold] = 2
        img[(img >= nonmetal_threshold) & (img < feed_threshold)] = 1
        return img

    def apply_unnormalization_and_binarization_on_components(self, components: np.ndarray) -> np.ndarray:
        components_unnormalized = self.unnormalize_principal_components(components)
        return self.apply_binarization_on_components(components_unnormalized)


class AntennaDataSet(torch.utils.data.Dataset):
    def __init__(self, antenna_folders: list[str], repr_mode: str, pca_wrapper: PCAWrapper, try_cache: bool):
        assert repr_mode in ['abs', 'rel', 'both'], 'Invalid dataset representation mode'
        self.repr_mode = repr_mode
        self.antenna_folders = antenna_folders
        self.len = len(antenna_folders)
        self.pca_wrapper = pca_wrapper
        self.try_cache = try_cache
        self.antenna_hw = (144, 200)  # self.antenna_hw = (20, 20)
        self.ant, self.embeddings, self.gam, self.rad, self.env = None, None, None, None, None
        self.ant_abs, self.env_abs = None, None
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
        embs = self.embeddings.detach().clone()
        self.embeddings = None
        if self.repr_mode == 'abs':
            return self.ant_abs, self.gam, self.rad, self.env_abs, antenna_name
        elif self.repr_mode == 'rel':
            return embs, self.gam, self.rad, self.env, antenna_name
        else:  # both
            return embs, self.ant_abs, self.gam, self.rad, self.env, self.env_abs, antenna_name

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
        self.env_abs = torch.tensor(self.env_abs).float()
        self.ant_abs = torch.tensor(self.ant_abs).float()

    def resize_antenna(self):
        h, w = self.antenna_hw
        return cv2.resize(self.ant, (w, h))

    def load_antenna(self, antenna_folder):
        self.ant = np.load(os.path.join(antenna_folder, 'antenna.npy'))
        self.gam = downsample_gamma(np.load(os.path.join(antenna_folder, 'gamma.npy'))[np.newaxis], rate=4).squeeze()
        self.rad = downsample_radiation(np.load(os.path.join(antenna_folder, 'radiation_directivity.npy'))[np.newaxis],
                                        rates=[4, 2]).squeeze()
        self.rad = self.clip_radiation(self.rad)
        self.gam = self.clip_gamma(self.gam)
        self.env = np.load(os.path.join(antenna_folder, 'environment.npy'))
        self.ant_abs = self.transform_ant_to_abs(self.ant, self.env)
        self.env_abs = self.transform_env_to_abs(self.env)
        if self.try_cache and os.path.exists(os.path.join(antenna_folder, 'embeddings.npy')):
            self.embeddings = np.load(os.path.join(antenna_folder, 'embeddings.npy'))

    @staticmethod
    def transform_ant_to_abs(ant: np.ndarray, env: np.ndarray) -> np.ndarray:
        ant_rel_og_repr = ant_to_dict_representation(torch.tensor(ant[np.newaxis]))[0]
        env_og_repr = env_to_dict_representation(torch.tensor(env[np.newaxis]))[0]
        ant_abs_og_repr = ant_rel2abs(ant_rel_og_repr, env_og_repr)
        ant_abs = np.array(list(ant_abs_og_repr.values()))
        return ant_abs

    @staticmethod
    def transform_env_to_abs(env: np.ndarray) -> np.ndarray:
        env_og_repr = env_to_dict_representation(torch.tensor(env[np.newaxis]))[0]
        env_abs_og_repr = model_rel2abs(env_og_repr)
        env_abs = np.array(list(env_abs_og_repr.values()))
        return env_abs

    @staticmethod
    def clip_radiation(radiation: np.ndarray):
        assert len(radiation.shape) == 3, 'Radiation shape is not 3D (channels, h, w)'
        sep_radiation = int(radiation.shape[0] / 2)
        radiation[:sep_radiation] = np.clip(radiation[:sep_radiation], -15, 5)
        return radiation

    @staticmethod
    def clip_gamma(gamma: np.ndarray):
        assert len(gamma.shape) == 1
        sep_gamma = int(gamma.shape[0] / 2)
        gamma[:sep_gamma] = np.clip(gamma[:sep_gamma], -20, 1e-9)
        return gamma

    def __reset_all(self):
        self.ant, self.embeddings, self.gam, self.rad, self.env = None, None, None, None, None


def filter_out_folder(folder_path):
    import time
    start = time.time()
    gamma_file = os.path.join(folder_path, 'gamma.npy')
    gamma = downsample_gamma(np.load(gamma_file)[np.newaxis], rate=4).squeeze()
    gamma_phase = gamma[gamma.shape[0] // 2:]
    num_minimas = find_local_minima(gamma_phase, 9)
    end = time.time()
    print(f'Time taken to find local minima: {end - start} seconds')
    if len(num_minimas) >= 4:
        print(f'Filtering out folder: {folder_path}')
        return True


def find_local_minima(arr, n):
    if n < 1:
        raise ValueError("n must be at least 1")

    arr = np.asarray(arr)  # Ensure the input is a NumPy array

    # Handle edge cases: if the array is too small
    if len(arr) <= 2 * n:
        return np.array([], dtype=int)

    # Iterate through the array and check neighbors
    minima_indices = []
    for i in range(n, len(arr) - n):
        # Extract left and right neighbors
        left_neighbors = arr[i - n:i]
        right_neighbors = arr[i + 1:i + n + 1]

        # Check if the current value is smaller than all neighbors
        if arr[i] < np.min(left_neighbors) and arr[i] < np.min(right_neighbors):
            minima_indices.append(i)

    return np.array(minima_indices)


class AntennaDataSetsLoader:
    def __init__(self, dataset_path: str, batch_size: int = None, repr_mode: str = 'abs',
                 pca: Optional[PCA] = None, split_ratio=None, try_cache=True):
        assert os.path.exists(dataset_path), f'Dataset path does not exist in {dataset_path}'
        if split_ratio is None:
            split_ratio = [0.8, 0.2, 0.0]  # [trn, val, tst]
        self.pca_wrapper = PCAWrapper(pca)
        self.split = split_ratio
        self.repr_mode = repr_mode
        self.trn_folders, self.val_folders, self.tst_folders = [], [], []
        self.split_data(dataset_path, split_ratio)
        self.batch_size = batch_size if batch_size is not None else len(self.trn_folders)
        self.trn_dataset = AntennaDataSet(self.trn_folders, repr_mode, self.pca_wrapper, try_cache)
        self.val_dataset = AntennaDataSet(self.val_folders, repr_mode, self.pca_wrapper, try_cache)
        self.tst_dataset = AntennaDataSet(self.tst_folders, repr_mode, self.pca_wrapper, try_cache)
        self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, batch_size=self.batch_size, drop_last=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, batch_size=self.batch_size)

    def split_data(self, dataset_path, split_ratio):
        filtered_folders_file = os.path.join(dataset_path, 'filtered_dataset_folders.pkl')
        if os.path.exists(filtered_folders_file):
            print('Loading filtered folders from cache')
            with open(filtered_folders_file, 'rb') as f:
                all_folders_basenames = pickle.load(f)
                all_folders = [os.path.join(dataset_path, f) for f in all_folders_basenames]
        else:
            print('filtered folders not found, not filtering')
            all_folders = sorted(glob.glob(os.path.join(dataset_path, '[0-9]*')))
        random.seed(42)
        random.shuffle(all_folders)
        trn_len = int(len(all_folders) * split_ratio[0])
        val_len = int(len(all_folders) * split_ratio[1])
        tst_len = len(all_folders) - trn_len - val_len
        self.trn_folders = all_folders[:trn_len]
        self.val_folders = all_folders[trn_len:trn_len + val_len]
        self.tst_folders = all_folders[trn_len + val_len:]

    def load_test_data(self, test_path):
        assert os.path.exists(test_path), f'Test path does not exist in {test_path}'
        self.tst_folders = sorted([f.path for f in os.scandir(test_path) if f.is_dir()])
        self.tst_dataset = AntennaDataSet(self.tst_folders, self.repr_mode, self.pca_wrapper, try_cache=False)
        self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, batch_size=self.batch_size)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
        else:
            self.scaler = standard_scaler()

    def fit(self, data: np.ndarray):
        assert self.scaler is not None, 'Scaler is not initialized'
        self.scaler.fit(data)
        print(f'Scaler fitted')

    def dump(self, path: str = None):
        dump_path = path if path is not None else self.path
        pickle.dump(self.scaler, open(dump_path, 'wb'))
        print(f'Fitted scaler saved in: {dump_path}')

    def fit_and_dump(self, data: np.ndarray, path: str = None):
        self.fit(data)
        self.dump(path)


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


def gamma_to_dB(gamma: torch.Tensor):
    gamma_copy = gamma.clone()  # Create a copy to avoid in-place modification
    gamma_copy[:, :int(gamma_copy.shape[1] / 2)] = gamma_mag_to_dB(gamma_copy[:, :int(gamma_copy.shape[1] / 2)])
    return gamma_copy

def gamma_mag_to_dB(gamma_mag: torch.Tensor):
    gamma_mag_dB = 20 * torch.log10(gamma_mag)
    return gamma_mag_dB

def gamma_to_linear(gamma: torch.Tensor):
    gamma_copy = gamma.clone()  # Create a copy to avoid in-place modification
    gamma_copy[:, :int(gamma_copy.shape[1] / 2)] = gamma_mag_to_linear(gamma_copy[:, :int(gamma_copy.shape[1] / 2)])
    return gamma_copy

def gamma_mag_to_linear(gamma_mag: torch.Tensor):
    gamma_mag_linear = 10 ** (gamma_mag / 20)
    return gamma_mag_linear

def radiation_to_dB(radiation: torch.Tensor):
    radiation_copy = radiation.clone()  # Create a copy to avoid in-place modification
    radiation_copy[:, :int(radiation_copy.shape[1] / 2)] = radiation_mag_to_dB(radiation_copy[:, :int(radiation_copy.shape[1] / 2)])
    return radiation_copy

def radiation_mag_to_dB(radiation_mag: torch.Tensor):
    radiation_mag_dB = 10 * torch.log10(radiation_mag)
    return radiation_mag_dB

def radiation_to_linear(radiation: torch.Tensor):
    radiation_copy = radiation.clone()  # Create a copy to avoid in-place modification
    radiation_copy[:, :int(radiation_copy.shape[1] / 2)] = radiation_mag_to_linear(radiation_copy[:, :int(radiation_copy.shape[1] / 2)])
    return radiation_copy

def radiation_mag_to_linear(radiation_mag: torch.Tensor):
    radiation_mag_linear = 10 ** (radiation_mag / 10)
    return radiation_mag_linear



def produce_gamma_stats(GT_gamma, predicted_gamma, dataset_type='linear', to_print=True):
    if dataset_type == 'linear':
        GT_gamma = gamma_to_dB(GT_gamma)
    if dataset_type == 'dB':
        pass
    if type(predicted_gamma) == tuple:
        GT_gamma, _ = GT_gamma
        predicted_gamma, _ = predicted_gamma
    predicted_gamma_mag, GT_gamma_mag = predicted_gamma[:, :int(predicted_gamma.shape[1] / 2)], GT_gamma[:, :int(
        GT_gamma.shape[1] / 2)]
    predicted_gamma_phase, GT_gamma_phase = predicted_gamma[:, int(predicted_gamma.shape[1] / 2):], GT_gamma[:, int(
        GT_gamma.shape[1] / 2):]
    diff_dB = torch.abs(predicted_gamma_mag - GT_gamma_mag)
    respective_diff = torch.where(GT_gamma_mag < -1.5, torch.div(diff_dB, torch.abs(GT_gamma_mag)) * 100, 0)
    avg_respective_diff = torch.mean(respective_diff[torch.nonzero(respective_diff, as_tuple=True)]).item()
    avg_diff = torch.mean(diff_dB, dim=1)
    max_diff = torch.max(diff_dB, dim=1)[0]
    # max_diff = torch.quantile(diff_dB,q=0.95,dim=1)
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


def polarized_to_total_radiation_mag(radiation: torch.Tensor):
    assert len(radiation.shape) == 4, "Radiation must be of shape (B, C, H, W)"
    rad_first_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(radiation[:, 0])**2 + radiation_mag_to_linear(radiation[:, 1])**2))
    rad_second_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(radiation[:, 2])**2 + radiation_mag_to_linear(radiation[:, 3])**2))
    rad_third_freq = radiation_mag_to_dB(torch.sqrt(radiation_mag_to_linear(radiation[:, 4])**2 + radiation_mag_to_linear(radiation[:, 5])**2))
    total_radiation = torch.stack((rad_first_freq, rad_second_freq, rad_third_freq), dim=1)
    return total_radiation


def produce_radiation_stats(predicted_radiation, gt_radiation, to_print=True):
    msssim = MSSSIM()
    if type(predicted_radiation) == tuple:
        _, predicted_radiation = predicted_radiation
        _, gt_radiation = gt_radiation
    sep = predicted_radiation.shape[1] // 2
    pred_rad_mag, gt_rad_mag = predicted_radiation[:, :sep], gt_radiation[:, :sep]
    #pred_rad_mag, gt_rad_mag = polarized_to_total_radiation_mag(pred_rad_mag), polarized_to_total_radiation_mag(gt_rad_mag)
    pred_rad_phase, gt_rad_phase = predicted_radiation[:, sep:], gt_radiation[:, sep:]
    abs_diff_mag = torch.abs(pred_rad_mag - gt_rad_mag)
    respective_diff = torch.where(torch.abs(gt_rad_mag) > 1.5, torch.div(abs_diff_mag, torch.abs(gt_rad_mag)) * 100, 0)
    respective_diff = torch.mean(respective_diff[torch.nonzero(respective_diff, as_tuple=True)]).item()
    diff_phase = pred_rad_phase - gt_rad_phase
    while len(torch.where(diff_phase > np.pi)[0]) > 0 or len(torch.where(diff_phase < -np.pi)[0]) > 0:
        diff_phase[torch.where(diff_phase > np.pi)] -= 2 * np.pi
        diff_phase[torch.where(diff_phase < -np.pi)] += 2 * np.pi
    max_diff_mag = torch.amax(abs_diff_mag, dim=(1, 2, 3))
    # max_diff_mag = torch.quantile(abs_diff_mag.reshape(abs_diff_mag.shape[0], -1), q=0.95, dim=1)
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
              , '%, mean_max_error_mag:', round(mean_max_error_mag, 4),
              ' dB, msssim_mag:', round(avg_msssim_mag, 4))
    return mean_abs_error_mag, max_diff_mag, msssim_vals


def produce_stats_all_dataset(gamma_stats: Union[List[Tuple], np.ndarray],
                              radiation_stats: Union[List[Tuple], np.ndarray]):
    print('--' * 20)
    gamma_stats_gathered = torch.tensor(gamma_stats)
    gamma_stats_mean = torch.nanmean(gamma_stats_gathered, dim=0).numpy()
    assert len(gamma_stats_mean) == 4, 'gamma stats mean should have 4 elements' \
                                       ' (avg mag, max mag, avg phase, max phase)'
    metrics_gamma_keys = [x + ' diff' for x in ['avg mag', 'max mag', 'avg phase', 'max phase']]
    stats_dict_gamma = dict(zip(metrics_gamma_keys, gamma_stats_mean))
    print(f'GAMMA STATS, averaged over entire dataset: {stats_dict_gamma}')
    radiation_stats_gathered = torch.tensor(radiation_stats)
    radiation_stats_mean = torch.nanmean(radiation_stats_gathered, dim=0).numpy()
    assert len(radiation_stats_mean) == 3, 'radiation stats mean should have 4 elements' \
                                           ' (avg mag, max mag, msssim)'
    metrics_rad_keys = [x + ' diff' for x in ['avg mag', 'max mag', 'msssim']]
    stats_dict_rad = dict(zip(metrics_rad_keys, radiation_stats_mean))
    print(f'RADIATION STATS, averaged over entire dataset: {stats_dict_rad}')
    print('--' * 20)
    pass


def save_antenna_mat(antenna: torch.Tensor, path: str, scaler: standard_scaler):
    import scipy.io as sio
    antenna = antenna.detach().cpu().numpy()
    antenna_unscaled = scaler.inverse(antenna)
    sio.savemat(path, {'antenna': antenna_unscaled})


def check_ant_validity(ant_parameters, model_parameters) -> int:
    assert int(model_parameters["type"]) in [3, 5], 'model parameters["type"] must be either 3 or 5.'
    if int(model_parameters['type']) == 3:
        Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters[
            'w'] / 2
              - model_parameters['feed_length'] / 2)
        Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
    else:
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sz'] - ant_parameters['w']
    wings = ['w1', 'w2', 'q1', 'q2']
    for key in ant_parameters:
        if ant_parameters[key] < 0: return 0
        if key != 'w' and ant_parameters[key] > 1: return 0
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
        if ant_parameters[f'{wing}y2'] < ant_parameters['w'] / Sy: return 0
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
        if np.abs(ant_parameters[f'{wing}z0'] - ant_parameters[f'{wing}z1']) <= ant_parameters['w'] / Sz: return 0
    if np.min([ant_parameters[f'q3z0'], ant_parameters[f'w3z0']]) > 0.2: return 0
    return 1


def model_rel2abs(model_parameters):
    model_parameters_abs = model_parameters.copy()
    assert int(model_parameters["type"]) in [3, 5], 'model parameters["type"] must be either 3 or 5.'
    if int(model_parameters['type']) == 3:
        axes = ['x', 'y', 'z']
        dimensions = ['width', 'height', 'length']
        elements = ['a', 'b', 'c', 'd']
        for e in elements:
            for i_axis, axis in enumerate(axes):
                model_parameters_abs[e + 'd' + axis] = model_parameters[e + 'd' + axis] * model_parameters[
                    dimensions[i_axis]]
                model_parameters_abs[e + 'r' + axis] = model_parameters[e + 'r' + axis] * model_parameters[e + 'd' + axis] * \
                                                       model_parameters[dimensions[i_axis]]
        model_parameters_abs['a'] = model_parameters['a'] * model_parameters['width']
        model_parameters_abs['b'] = model_parameters['b'] * model_parameters['height']
        model_parameters_abs['c'] = model_parameters['c'] * model_parameters['height']
    else:
        model_parameters_abs['Lz'] = model_parameters['Sz'] * model_parameters_abs['Lz']
        model_parameters_abs['Ly'] = model_parameters['Sy'] * model_parameters_abs['Ly']
    return model_parameters_abs


def ant_to_dict_representation(ant: torch.Tensor):
    assert ant.ndim == 2, 'Antenna tensor should have 2 dimensions (batch, features)'
    ant_path = os.path.join(EXAMPLE_FOLDER, 'ant_parameters.pickle')
    with open(ant_path, 'rb') as f:
        example = pickle.load(f)
    all_ant_dicts = []
    ant = ant.clone().detach().cpu().numpy()
    for i in range(ant.shape[0]):
        ant_i = np.round(ant[i], 2)
        ant_i_dict = {key: val for key, val in zip(example.keys(), ant_i)}
        all_ant_dicts.append(ant_i_dict)
    return np.array(all_ant_dicts)


def env_to_dict_representation(env: torch.Tensor):
    assert env.ndim == 2, 'Environment tensor should have 2 dimensions (batch, features)'
    env_path = os.path.join(EXAMPLE_FOLDER, 'model_parameters.pickle')
    with open(env_path, 'rb') as f:
        example = pickle.load(f)
    all_env_dicts = []
    env = env.clone().detach().cpu().numpy()
    for i in range(env.shape[0]):
        env_i = np.round(np.append([MODEL_TYPE], env[i]), 2)
        env_i_dict = {key: val for key, val in zip(example.keys(), env_i)}
        all_env_dicts.append(env_i_dict)
    return np.array(all_env_dicts)


def ant_rel2abs(ant_parameters: dict, model_parameters: dict):
    ant_parameters_abs = ant_parameters.copy()
    assert int(model_parameters["type"]) in [3, 5], 'model parameters["type"] must be either 3 or 5.'
    if int(model_parameters["type"]) == 3:
        Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters['w'] / 2
              - model_parameters['feed_length'] / 2)
        Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
    else:
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sz'] - ant_parameters['w']
    for key, value in ant_parameters.items():
        if len(key) == 4:
            if key[2] == 'z':
                ant_parameters_abs[key] = np.round(value * Sz, decimals=2)
            if key[2] == 'y':
                ant_parameters_abs[key] = np.round(value * Sy, decimals=2)
        if key == 'fx':
            ant_parameters_abs[key] = np.round(value * Sy, decimals=2)
    return ant_parameters_abs


def ant_abs2rel(ant_parameters_abs: dict, model_parameters: dict):
    ant_parameters_rel = ant_parameters_abs.copy()
    assert int(model_parameters["type"]) in [3, 5], 'model parameters["type"] must be either 3 or 5.'
    if int(model_parameters["type"]) == 3:
        Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters_abs[
            'w'] / 2
              - model_parameters['feed_length'] / 2)
        Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters_abs['w']
    else:
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sz'] - ant_parameters['w']
    for key, value in ant_parameters_abs.items():
        if len(key) == 4:
            if key[2] == 'z':
                ant_parameters_rel[key] = np.round(value / Sz, decimals=2)
            if key[2] == 'y':
                ant_parameters_rel[key] = np.round(value / Sy, decimals=2)
        if key == 'fx':
            ant_parameters_rel[key] = np.round(value / Sy, decimals=2)
    return ant_parameters_rel


class AntValidityFunction:
    def __init__(
        self,
        sample_path: str,
        ant_scaler: ScalerManager
    ) -> None:
        self.path = sample_path
        self.ant_scaler = ant_scaler

    def __call__(self, ant: torch.Tensor) -> bool:
        env_og_rel_repr = env_to_dict_representation(
            torch.tensor(np.load(os.path.join(self.path, 'environment.npy'))[np.newaxis]))[0]
        ant_abs = self.ant_scaler.scaler.inverse(ant)
        ant_og_abs_repr = ant_to_dict_representation(ant_abs)[0]
        ant_og_rel_repr = ant_abs2rel(ant_og_abs_repr, env_og_rel_repr)
        return bool(check_ant_validity(ant_og_rel_repr, env_og_rel_repr))


class data_linewidth_plot:
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
    assert int(model_parameters["type"]) in [3, 5], 'model parameters["type"] must be either 3 or 5.'
    if int(model_parameters["type"]) == 3:
        Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters['w'] / 2
              - model_parameters['feed_length'] / 2)
        Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
    else:
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sz'] - ant_parameters['w']

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
