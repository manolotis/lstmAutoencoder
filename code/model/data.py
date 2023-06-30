import random
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


def angle_to_range(yaw):
    yaw = (yaw - np.pi) % (2 * np.pi) - np.pi
    return yaw


def normalize(data, config, split="train"):
    features = tuple(config[split]["data_config"]["dataset_config"]["input_data"])
    if features == ("xy", "yaw", "v_xy", "width", "length", "valid"):
        normalization_means = {
            "target/history/lstm_data": np.array([
                -3.037965933131866,  # x
                0.005443134484205298,  # y
                -0.0033017450976257842,  # yaw
                6.174656202228034,  # vx
                -0.02157458059128233,  # vy
                1.9719390214387036,  # width
                4.374759175074724,  # length
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        normalization_stds = {
            "target/history/lstm_data": np.array([
                3.793132077322926,
                0.16012741081460383,
                0.12045736633428678,
                5.638115207720868,
                0.3821443614578785,
                0.4693856644817378,
                1.507592546518198,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        }
    else:
        raise Exception("Wrong features set")
    keys = ['target/history/lstm_data']
    for k in keys:
        data[k] = (data[k] - normalization_means[k]) / (normalization_stds[k] + 1e-6)
        data[k].clamp_(-15, 15)

    data[f"target/history/lstm_data"] *= data[f"target/history/valid"]

    return data


def dict_to_cuda(d):
    passing_keys = set([
        'target/history/lstm_data', 'target/history/lstm_data_diff',
        'other/history/lstm_data', 'other/history/lstm_data_diff',
        'target/history/mcg_input_data', 'other/history/mcg_input_data',
        'other_agent_history_scatter_idx', 'road_network_scatter_idx',
        'other_agent_history_scatter_numbers', 'road_network_scatter_numbers',
        'batch_size',
        'road_network_embeddings',
        'target/future/xy', 'target/future/valid'])
    for k in d.keys():
        if k not in passing_keys:
            continue
        v = d[k]
        if not isinstance(v, torch.Tensor):
            continue
        d[k] = d[k].cuda()


class LSTMAutoencoderDataset(Dataset):

    def __init__(self, config, noise_config=None):
        self._data_path = config["data_path"]
        self._config = config
        self._noise_config = noise_config
        self._training = noise_config["training"]
        print("noise config: ", noise_config)
        files = os.listdir(self._data_path)
        self._files = [os.path.join(self._data_path, f) for f in files]

        print("Len files before adding noisy: ", len(self._files))
        if noise_config is not None and self._training:
            # Do not add extra files for perturbations unless we are in training mode
            self._add_noisy_files()
        print("Len files after adding noisy: ", len(self._files))

        self._files = sorted(self._files)

        assert config["n_shards"] > 0

        if "max_length" in config:
            self._files = self._files[:config["max_length"]]

            if config["n_shards"] > 1:
                raise ValueError("Limiting training data with limit>0 and n_shards>1. Choose one or the other.")

        if config["n_shards"] > 1:
            self._files = [file for (i, file) in enumerate(self._files) if i % config["n_shards"] == 0]

        random.shuffle(self._files)

        assert len(self._files) > 0

    def __len__(self):
        return len(self._files)

    def _add_noisy_files(self):
        if self._noise_config["noisy_heading"]:  # load from pre-rendered folder
            files_noisy = os.listdir(self._noise_config["data_path"])
            files_noisy = [os.path.join(self._noise_config["data_path"], f) for f in files_noisy]
            self._files.extend(files_noisy)

        if self._noise_config["exclude_road"]:  # (re)load from same folder with flag in path
            files_noisy = os.listdir(self._data_path)
            files_noisy = [os.path.join(self._data_path, f + "exclude_road") for f in files_noisy]
            self._files.extend(files_noisy)

        if self._noise_config["hide_target_past"]:  # (re)load from same folder with flag in path
            files_noisy = os.listdir(self._data_path)
            files_noisy = [os.path.join(self._data_path, f + "hide_target_past") for f in files_noisy]
            self._files.extend(files_noisy)

    def _compute_agent_type_and_is_sdc_ohe(self, data, subject):
        I = np.eye(5)
        agent_type_ohe = I[np.array(data[f"{subject}/agent_type"])]
        is_sdc = np.array(data[f"{subject}/is_sdc"]).reshape(-1, 1)
        ohe_data = np.concatenate([agent_type_ohe, is_sdc], axis=-1)[:, None, :]
        ohe_data = np.repeat(ohe_data, data["target/history/xy"].shape[1], axis=1)
        return ohe_data

    def _add_length_width(self, data):
        data["target/history/length"] = \
            data["target/length"].reshape(-1, 1, 1) * np.ones_like(data["target/history/yaw"])
        data["target/history/width"] = \
            data["target/width"].reshape(-1, 1, 1) * np.ones_like(data["target/history/yaw"])

        data["other/history/length"] = \
            data["other/length"].reshape(-1, 1, 1) * np.ones_like(data["other/history/yaw"])
        data["other/history/width"] = \
            data["other/width"].reshape(-1, 1, 1) * np.ones_like(data["other/history/yaw"])
        return data

    def _hide_target_history(self, np_data):
        # mask all timesteps except the latest one
        for key in np_data.keys():
            if "target/history" in key:
                np_data[key][:, :-1, :] = -1
        np_data["target/history/valid"][:, :-1, :] = 0
        return np_data

    def _compute_lstm_input_data(self, data):
        keys_to_stack = self._config["input_data"]
        for subject in ["target"]:
            agent_type_ohe = self._compute_agent_type_and_is_sdc_ohe(data, subject)
            data[f"{subject}/history/lstm_data"] = np.concatenate(
                [data[f"{subject}/history/{k}"] for k in keys_to_stack] + [agent_type_ohe], axis=-1)
            data[f"{subject}/history/lstm_data"] *= data[f"{subject}/history/valid"]

        return data

    def _prepare_and_get_path(self, idx):
        if not self._training:
            return self._files[idx]

        file_to_load = self._files[idx]
        if "exclude_road" in file_to_load:
            path = file_to_load.replace("exclude_road", "")
            self._noise_config["exclude_road"] = True
            self._noise_config["hide_target_past"] = False
        elif "hide_target_past" in file_to_load:
            path = file_to_load.replace("hide_target_past", "")
            self._noise_config["exclude_road"] = False
            self._noise_config["hide_target_past"] = True
        else:
            # load normally
            path = file_to_load
            self._noise_config["exclude_road"] = False
            self._noise_config["hide_target_past"] = False

        return path

    def __getitem__(self, idx):
        path = self._prepare_and_get_path(idx)
        np_data = dict(np.load(path, allow_pickle=True))
        np_data["scenario_id"] = np_data["scenario_id"].item()
        np_data["filename"] = self._files[idx]
        np_data["target/history/yaw"] = angle_to_range(np_data["target/history/yaw"])
        np_data["other/history/yaw"] = angle_to_range(np_data["other/history/yaw"])
        np_data = self._add_length_width(np_data)
        if self._noise_config["hide_target_past"]:
            np_data = self._hide_target_history(np_data)
        np_data = self._compute_lstm_input_data(np_data)

        return np_data

    @staticmethod
    def collate_fn(batch):
        batch_keys = [k for k in batch[0].keys() if k != "yaw_original"]
        result_dict = {k: [] for k in batch_keys}

        for sample_num, sample in enumerate(batch):
            for k in batch_keys:
                if not isinstance(sample[k], str) and len(sample[k].shape) == 0:
                    result_dict[k].append(sample[k].item())
                else:
                    result_dict[k].append(sample[k])

        for k, v in result_dict.items():
            if not isinstance(v[0], np.ndarray):
                continue
            result_dict[k] = torch.Tensor(np.concatenate(v, axis=0))

        result_dict["batch_size"] = len(batch)
        return result_dict


def get_dataloader(config):
    dataset = LSTMAutoencoderDataset(config["dataset_config"], config["noise_config"])
    dataloader = DataLoader(
        dataset, collate_fn=LSTMAutoencoderDataset.collate_fn, **config["dataloader_config"])
    return dataloader
