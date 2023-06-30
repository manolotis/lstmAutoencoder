from multipathPP.code.utils.train_utils import get_config, parse_arguments
from lstmAutoencoder.code.model.data import LSTMAutoencoderDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader

args = parse_arguments()
config = get_config(args)


def get_dataloader(config):
    dataset = LSTMAutoencoderDataset(config["dataset_config"])
    dataloader = DataLoader(dataset, collate_fn=LSTMAutoencoderDataset.collate_fn, **config["dataloader_config"])
    return dataloader


dataloader = get_dataloader(config["data_config"])

all_values = {
    "x": [],
    "y": [],
    "valid": [],
}

max_trajs = 500

for data in tqdm(dataloader):
    n_agents = data["target/history/xy"].shape[0]

    all_values["x"].extend(data["target/history/xy"].numpy()[..., 0].flatten().tolist())
    all_values["y"].extend(data["target/history/xy"].numpy()[..., 1].flatten().tolist())
    all_values["valid"].extend((data["target/history/valid"].numpy().flatten() > 0).tolist())

    all_values["x"].extend(data["target/future/xy"].numpy()[..., 0].flatten().tolist())
    all_values["y"].extend(data["target/future/xy"].numpy()[..., 1].flatten().tolist())
    all_values["valid"].extend((data["target/future/valid"].numpy().flatten() > 0).tolist())

    max_trajs -= n_agents
    if max_trajs < 0:
        break

all_values["x"] = np.array(all_values["x"])
all_values["y"] = np.array(all_values["y"])
all_values["valid"] = np.array(all_values["valid"])

plt.figure()
plt.scatter(all_values["x"][all_values["valid"]], all_values["y"][all_values["valid"]], s=1, alpha=0.1)
plt.tight_layout()
plt.show()

print("mean x", all_values["x"][all_values["valid"]].mean())
print("std x", all_values["x"][all_values["valid"]].mean())

print("mean y", all_values["y"][all_values["valid"]].mean())
print("std y", all_values["y"][all_values["valid"]].mean())
