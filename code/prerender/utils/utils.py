# ToDo: Move to waymo_utils repo
import yaml
from yaml import Loader
import numpy as np

yaml.warnings({'YAMLLoadWarning': False})


def data_to_numpy(data):
    for k, v in data.items():
        data[k] = v.numpy()
    data["parsed"] = True


def get_config(args):
    path = args.config
    with open(path, 'r') as stream:
        config = yaml.load(stream, Loader)

    # If provided in command line, override defaults

    keys = [
        "data_path",
        "n_shards",
        "shard_id",
        "data_path",
        "output_path",
    ]

    for key in keys:
        if key in config:
            config[key] = config[key] if getattr(args, key) is None else getattr(args, key)

    return config


def filter_valid(item, valid_array):
    return item[valid_array.flatten() > 0]


def get_filter_valid_agent_history():
    result = []
    key_with_different_timezones = ["x", "y", "speed", "bbox_yaw", "valid"]
    common_keys = [
        "state/id", "state/is_sdc", "state/type", "state/current/width", "state/current/length"]
    for key in key_with_different_timezones:
        for zone in ["past", "current", "future"]:
            result.append(f"state/{zone}/{key}")
    result.extend(common_keys)
    return result


def get_normalize_data():
    return {
        "target": {
            "xy": {
                "mean": np.array([[[-3.0173979, 0.00575967]]]),
                "std": np.array([[[3.7542882, 0.11941358]]])},
            "yaw": {
                "mean": np.array([[[0.00815599]]]),
                "std": np.array([[[1.0245908]]])},
            "speed": {
                "mean": np.array([[[6.1731253]]]),
                "std": np.array([[[5.53667]]])}},
        "other": {
            "xy": {
                "mean": np.array([[[9.855061, 2.6597235]]]),
                "std": np.array([[[44.58452, 34.069477]]])},
            "yaw": {
                "mean": np.array([[[1.6482836]]]),
                "std": np.array([[[3.7098966]]])},
            "speed": {
                "mean": np.array([[[2.5248919]]]),
                "std": np.array([[[4.806048]]])}
        },
    }
