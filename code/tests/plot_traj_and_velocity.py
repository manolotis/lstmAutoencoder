from multipathPP.code.utils.train_utils import get_config, parse_arguments
from lstmAutoencoder.code.model.data import LSTMAutoencoderDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

args = parse_arguments()
config = get_config(args)


def get_dataloader(config):
    dataset = LSTMAutoencoderDataset(config["dataset_config"], config["noise_config"])
    dataloader = DataLoader(dataset, collate_fn=LSTMAutoencoderDataset.collate_fn, **config["dataloader_config"])
    return dataloader


dataloader = get_dataloader(config["data_config"])

all_values = {
    "x": [],
    "y": [],
    "valid": [],
}

max_trajs = 3

for data in tqdm(dataloader):
    n_agents = data["target/history/xy"].shape[0]

    # print(data.keys())
    print(data["target/agent_type"])

    i = 14
    timezone = "history"
    X = data[f"target/{timezone}/xy"].numpy()[i, :, 0][data[f"target/{timezone}/valid"][i].flatten() > 0]
    Y = data[f"target/{timezone}/xy"].numpy()[i, :, 1][data[f"target/{timezone}/valid"][i].flatten() > 0]
    U = data[f"target/{timezone}/v_xy"].numpy()[i, :, 0][data[f"target/{timezone}/valid"][i].flatten() > 0] / 10.0
    V = data[f"target/{timezone}/v_xy"].numpy()[i, :, 1][data[f"target/{timezone}/valid"][i].flatten() > 0] / 10.0
    print(data[f"target/{timezone}/v_xy"][i] / 10)
    # Creating plot
    plt.quiver(X, Y, U, V, color='b', units='xy', scale=1)
    # plt.quiver(X, Y, U, V, color='b')
    plt.scatter(X, Y)

    # Show plot with grid
    plt.grid()
    plt.show()

    break
