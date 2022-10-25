import sys

sys.path.append("/home/manolotis/sandbox/robustness_benchmark/")
import copy

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
import subprocess
from model.lstm_encoder_decoder import LSTMEncoderDecoder
from model.data import get_dataloader, dict_to_cuda, normalize
import os
import glob
import random
from utils.predict_utils import parse_arguments, get_config
import numpy as np
from tqdm import tqdm

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

args = parse_arguments()
config = get_config(args)


def get_best_checkpoint(path):
    list_of_files = glob.glob(f'{path}/*')
    for f in list_of_files:
        if "best" in f:
            return f
    return None


try:
    models_path = os.path.join(config["model"]["path"], config["model"]["name"])
    # checkpoint_path = get_last_checkpoint(models_path)
    checkpoint_path = get_best_checkpoint(models_path)
    test_dataloader = get_dataloader(config["test"]["data_config"])
    model = LSTMEncoderDecoder(config["model"])
    model.cuda()
    if checkpoint_path is not None:
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])
        num_steps = torch.load(checkpoint_path)["num_steps"]
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("LOADED ", checkpoint_path, "with N params=", params)
    else:
        raise ValueError(f"Checkpoint was None when trying to load from: {checkpoint_path}")
    model.eval()


except Exception as e:
    # ToDo: deal with this
    raise e


def generate_filename(scene_data, agent_index):
    scenario_id = scene_data["scenario_id"][agent_index]
    agent_id = scene_data["agent_id"][agent_index]
    agent_type = scene_data["target/agent_type"][agent_index]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}.npz"


savefolder = os.path.join(config["test"]["output_config"]["out_path"], config["model"]["name"])

if not os.path.exists(savefolder):
    os.makedirs(savefolder, exist_ok=True)

for data in tqdm(test_dataloader):
    if config["test"]["normalize"]:
        data_original = copy.deepcopy(data)
        data = normalize(data, config, split="test")
    else:
        data_original = data
    dict_to_cuda(data)
    coordinates = model(data).unsqueeze(1).cuda()

    means = torch.Tensor([23.010058080655632, 0.5563976456234273]).cuda()
    stds = torch.Tensor([26.74993430078498, 9.108507110604085]).cuda()

    coordinates = coordinates * stds + means
    coordinates = coordinates.detach().cpu()



    for agent_index, agent_id in enumerate(data["agent_id"]):
        filename = generate_filename(data, agent_index)
        savedata = {
            "scenario_id": data["scenario_id"][agent_index],
            "agent_id": data["agent_id"][agent_index],
            "agent_type": data["target/agent_type"][agent_index].flatten(),
            "coordinates": coordinates[agent_index],
            "probabilities": None,
            "target/history/xy": data_original["target/history/xy"][agent_index],
            "target/future/xy": data_original["target/future/xy"][agent_index],
            "target/history/valid": data_original["target/history/valid"][agent_index],
            "target/future/valid": data_original["target/future/valid"][agent_index]
        }

        # print("target/history/xy", data_original["target/history/xy"])
        # print("target/future/xy", data_original["target/future/xy"])
        # print("target/future/valid", data_original["target/future/valid"])
        # exit()

        np.savez_compressed(os.path.join(savefolder, filename), **savedata)
