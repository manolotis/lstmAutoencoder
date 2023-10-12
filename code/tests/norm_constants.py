from multipathPP.code.utils.train_utils import get_config, parse_arguments
from lstmAutoencoder.code.model.data import LSTMAutoencoderDataset
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

args = parse_arguments()
config = get_config(args)


def get_dataloader(config):
    dataset = LSTMAutoencoderDataset(config["dataset_config"])
    dataloader = DataLoader(dataset, collate_fn=LSTMAutoencoderDataset.collate_fn, **config["dataloader_config"])
    return dataloader


dataloader = get_dataloader(config["data_config"])

all_values = {
    "target/width": [],
    "target/length": [],
    "target/history/yaw": [],
    "target/history/x": [],
    "target/history/y": [],
    "target/history/vx": [],
    "target/history/vy": [],
    "target/history/valid": [],
    "target/future/x": [],
    "target/future/y": [],
    "target/future/valid": [],
}

for data in tqdm(dataloader):
    all_values["target/length"].extend(data["target/length"])
    all_values["target/width"].extend(data["target/width"])

    all_values["target/history/x"].extend(data["target/history/xy"].numpy()[..., 0].flatten().tolist())
    all_values["target/history/y"].extend(data["target/history/xy"].numpy()[..., 1].flatten().tolist())
    all_values["target/history/vx"].extend(data["target/history/v_xy"].numpy()[..., 0].flatten().tolist())
    all_values["target/history/vy"].extend(data["target/history/v_xy"].numpy()[..., 1].flatten().tolist())
    all_values["target/history/yaw"].extend(data["target/history/yaw"].numpy().flatten().tolist())
    all_values["target/history/valid"].extend((data["target/history/valid"].numpy().flatten() > 0).tolist())

    all_values["target/future/x"].extend(data["target/future/xy"].numpy()[..., 0].flatten().tolist())
    all_values["target/future/y"].extend(data["target/future/xy"].numpy()[..., 1].flatten().tolist())
    all_values["target/future/valid"].extend((data["target/future/valid"].numpy().flatten() > 0).tolist())

for k, values in all_values.items():
    values = np.array(values)
    timezone = None
    if 'future' in k:
        timezone = 'future'
    if 'history' in k:
        timezone = 'history'

    if timezone is None:
        print(f"{k} mean: {np.array(values).mean()}")
        print(f"{k} std: {np.array(values).std()}")
    else:
        print(f"{k} mean: {np.array(values[all_values[f'target/{timezone}/valid']]).mean()}")
        print(f"{k} std: {np.array(values[all_values[f'target/{timezone}/valid']]).std()}")
