import tensorflow as tf
import os
import numpy as np
from .renderer import LSTMAutoencoderRenderer
from .utils import data_to_numpy
import argparse


def get_renderer(renderer_name, renderer_config):
    if renderer_name == "LSTMAutoencoderRenderer":
        return LSTMAutoencoderRenderer(renderer_config)
    raise Exception(f"Unknown visualizer {renderer_name}")


def get_renderers(renderers_config):
    renderers = []
    for renderer in renderers_config:
        renderers.append(get_renderer(renderer["renderer_name"], renderer["renderer_config"]))
    return renderers


def create_dataset(datapath, n_shards, shard_id):
    files = os.listdir(datapath)
    dataset = tf.data.TFRecordDataset(
        [os.path.join(datapath, f) for f in files], num_parallel_reads=1
    )
    if n_shards > 1:
        dataset = dataset.shard(n_shards, shard_id)
    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=False, help="Path to raw data")
    parser.add_argument("--output-path", type=str, required=False, help="Path to save data")
    parser.add_argument("--n-jobs", type=int, required=False, help="Number of threads")
    parser.add_argument(
        "--n-shards", type=int, required=False, help="Use `1/n_shards` of full dataset")
    parser.add_argument(
        "--shard-id", type=int, required=False, help="Take shard with given id")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    return args


def generate_filename(scene_data):
    scenario_id = scene_data["scenario_id"]
    agent_id = scene_data["agent_id"]
    agent_type = scene_data["target/agent_type"]
    return f"scid_{scenario_id}__aid_{agent_id}__atype_{agent_type.item()}.npz"


def merge_and_save(renderers, data, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    data_to_numpy(data)
    preprocessed_dicts = [renderer.render(data) for renderer in renderers]
    for scene_number in range(len(preprocessed_dicts[0])):
        scene_data = {}
        for renderer_number in range(len(preprocessed_dicts)):
            scene_data.update(preprocessed_dicts[renderer_number][scene_number])
        np.savez_compressed(os.path.join(output_path, generate_filename(scene_data)), **scene_data)
