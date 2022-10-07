from torch.utils.data.dataset import T_co

from multipathPP.code.utils.train_utils import get_config, parse_arguments
# from multipathPP.code.model.data import get_dataloader
from lstmAutoencoder.code.model.data import LSTMAutoencoderDataset
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import random
import os

args = parse_arguments()
config = get_config(args)


# dataloader = get_dataloader(config["train"]["data_config"])


def get_dataloader(config):
    dataset = LSTMAutoencoderDataset(config["dataset_config"])
    dataloader = DataLoader(dataset, collate_fn=LSTMAutoencoderDataset.collate_fn, **config["dataloader_config"])
    return dataloader


dataloader = get_dataloader(config["data_config"])

all_values = {}

for data in tqdm(dataloader):

    keys = [
        "target/width",
        "target/length",
        "target/history/yaw",
        "target/future/yaw",
    ]

    keys2 = [  # (b,11,2)
        "target/history/xy",
        "target/history/v_xy",
        "target/future/xy",
        # "target/future/v_xy",
    ]

    for k in keys:
        if k not in all_values:
            all_values[k] = []

        try:
            all_values[k].extend(data[k].flatten().tolist())
        except AttributeError:
            data[k] = np.array(data[k])
            all_values[k].extend(data[k].flatten().tolist())

    for k in keys2:
        if k not in all_values:
            all_values[k+"_x"] = []
            all_values[k+"_y"] = []

        all_values[k+"_x"].extend(data[k][..., 0].flatten().tolist())
        all_values[k+"_y"].extend(data[k][..., 1].flatten().tolist())


for k, values in all_values.items():
    print(f"{k} mean: {np.array(all_values[k]).mean()}")
    print(f"{k} std: {np.array(all_values[k]).std()}")
