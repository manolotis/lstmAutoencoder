# ToDo: Move to waymo_utils repo
from abc import ABC, abstractmethod
import numpy as np
from .utils import filter_valid, get_filter_valid_agent_history, get_normalize_data


class Renderer(ABC):
    @abstractmethod
    def render(self, data):
        pass


class TargetAgentFilteringPolicy:
    def __init__(self, config):
        self._config = config

    def _get_only_interesting_agents(self, data, i):
        return data["state/tracks_to_predict"][i] > 0

    def _get_only_fully_available_agents(self, data, i):
        full_validity = np.concatenate(
            [
                data["state/past/valid"],
                data["state/current/valid"],
                data["state/future/valid"]
            ], axis=-1)
        n_timestamps = full_validity.shape[-1]
        n_valid_timestamps = full_validity.sum(axis=-1)
        return n_valid_timestamps[i] == n_timestamps
        # return np.ones_like(n_valid_timestamps) * (n_valid_timestamps == n_timestamps)

    def _get_interesting_and_fully_available_agents(self, data, i):
        interesting = self._get_only_interesting_agents(data, i)
        fully_valid = self._get_only_fully_available_agents(data, i)
        return interesting and fully_valid

    def _get_fully_available_agents_without_interesting(self, data, i):
        interesting = self._get_only_interesting_agents(data, i)
        fully_valid = self._get_only_fully_available_agents(data, i)
        return fully_valid and not interesting

    def allow(self, data, i):
        if self._config["policy"] == "interesting":
            return self._get_only_interesting_agents(data, i)
        if self._config["policy"] == "fully_available":
            return self._get_only_fully_available_agents(data, i)
        if self._config["policy"] == "interesting_and_fully_available":
            return self._get_interesting_and_fully_available_agents(data, i)
        if self._config["policy"] == "fully_available_agents_without_interesting":
            return self._get_fully_available_agents_without_interesting(data, i)
        raise Exception(f"Unknown agent filtering policy {self._config['policy']}")


class LSTMAutoencoderRenderer(Renderer):
    def __init__(self, config):
        self._config = config
        self._target_agent_filter = TargetAgentFilteringPolicy(self._config["agent_filtering"])

    def _select_agents_with_any_validity(self, data):
        return data["state/current/valid"].sum(axis=-1) + \
            data["state/future/valid"].sum(axis=-1) + data["state/past/valid"].sum(axis=-1)

    def _preprocess_data(self, data):
        agents_with_any_validity_selector = self._select_agents_with_any_validity(data)
        for key in get_filter_valid_agent_history():
            data[key] = filter_valid(data[key], agents_with_any_validity_selector)

    def _split_past_and_future(self, data, key):
        history = np.concatenate(
            [data[f"state/past/{key}"], data[f"state/current/{key}"]], axis=1)[..., None]
        future = data[f"state/future/{key}"][..., None]
        return history, future

    def _prepare_agent_history(self, data):
        # (n_agents, 11, 2)
        preprocessed_data = {}
        preprocessed_data["history/xy"] = np.array([
            np.concatenate([data["state/past/x"], data["state/current/x"]], axis=1),
            np.concatenate([data["state/past/y"], data["state/current/y"]], axis=1)
        ]).transpose(1, 2, 0)

        preprocessed_data["history/v_xy"] = np.array([
            np.concatenate([data["state/past/velocity_x"], data["state/current/velocity_x"]], axis=1),
            np.concatenate([data["state/past/velocity_y"], data["state/current/velocity_y"]], axis=1)
        ]).transpose(1, 2, 0)

        # (n_agents, 80, 2)
        preprocessed_data["future/xy"] = np.array(
            [data["state/future/x"], data["state/future/y"]]).transpose(1, 2, 0)
        preprocessed_data["future/v_xy"] = np.array(
            [data["state/future/velocity_x"], data["state/future/velocity_y"]]).transpose(1, 2, 0)

        # (n_agents, 11, 1)
        for key in ["bbox_yaw", "valid"]:
            preprocessed_data[f"history/{key}"], preprocessed_data[f"future/{key}"] = \
                self._split_past_and_future(data, key)
        for key in ["state/id", "state/is_sdc", "state/type", "state/current/width",
                    "state/current/length"]:
            preprocessed_data[key.split('/')[-1]] = data[key]
        preprocessed_data["scenario_id"] = data["scenario/id"]
        return preprocessed_data

    def _transfrom_to_agent_coordinate_system(self, coordinates, shift, yaw):
        # coordinates
        # dim 0: number of agents / number of segments for road network
        # dim 1: number of history points / (start_point, end_point) for segments
        # dim 2: x, y
        yaw = -yaw
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array(((c, -s), (s, c))).reshape(2, 2)
        transformed = np.matmul((coordinates - shift), R.T)
        return transformed

    def _compute_closest_point_of_segment(self, segments):
        # This method works only with road segments in agent-related coordinate system
        assert len(segments.shape) == 3
        assert segments.shape[1] == segments.shape[2] == 2
        A, B = segments[:, 0, :], segments[:, 1, :]
        M = B - A
        t = (-A * M).sum(axis=-1) / ((M * M).sum(axis=-1) + 1e-6)
        clipped_t = np.clip(t, 0, 1)[:, None]
        closest_points = A + clipped_t * M
        return closest_points

    def _normalize_tensor(self, tensor, mean, std):
        if not self._config["normalize"]:
            return tensor
        raise Exception("Normalizing here is really not what you want. Please use normalization from model.data")
        return (tensor - mean) / (std + 1e-6)

    def _normalize(self, tensor, i, key):
        target_data = tensor[i][None,]
        other_data = np.delete(tensor, i, axis=0)
        if not self._config["normalize"]:
            return target_data, other_data
        target_data = self._normalize_tensor(target_data, **get_normalize_data()["target"][key])
        other_data = self._normalize_tensor(other_data, **get_normalize_data()["other"][key])
        return target_data, other_data

    def render(self, data):
        array_of_scene_data_dicts = []
        self._preprocess_data(data)

        agent_history_info = self._prepare_agent_history(data)
        for i in range(agent_history_info["history/xy"].shape[0]):
            if not self._target_agent_filter.allow(data, i):
                continue

            current_agent_scene_shift = agent_history_info["history/xy"][i][-1]

            if self._config["noisy_heading"]:
                agent_history_info["history/bbox_yaw"][i][-1] += np.pi / 2

            current_agent_scene_yaw = agent_history_info["history/bbox_yaw"][i][-1]

            current_scene_agents_coordinates_history = self._transfrom_to_agent_coordinate_system(
                agent_history_info["history/xy"],
                current_agent_scene_shift,
                current_agent_scene_yaw
            )
            current_scene_agents_coordinates_future = self._transfrom_to_agent_coordinate_system(
                agent_history_info["future/xy"],
                current_agent_scene_shift,
                current_agent_scene_yaw
            )

            current_scene_agents_velocities_history = self._transfrom_to_agent_coordinate_system(
                agent_history_info["history/v_xy"],
                0,
                current_agent_scene_yaw
            )

            current_scene_agents_yaws_history = agent_history_info["history/bbox_yaw"] - current_agent_scene_yaw
            current_scene_agents_yaws_future = agent_history_info["future/bbox_yaw"] - current_agent_scene_yaw

            (current_scene_target_agent_coordinates_history,
             current_scene_other_agents_coordinates_history) = \
                self._normalize(
                    current_scene_agents_coordinates_history, i, "xy"
                )
            (current_scene_target_agent_velocities_history,
             current_scene_other_agents_velocities_history) = \
                self._normalize(
                    current_scene_agents_velocities_history, i, "v_xy"
                )

            current_scene_target_agent_yaws_history, current_scene_other_agents_yaws_history = \
                self._normalize(current_scene_agents_yaws_history, i, "yaw")

            scene_data = {
                "shift": current_agent_scene_shift[None,],
                "yaw": current_agent_scene_yaw,
                "scenario_id": agent_history_info["scenario_id"].item().decode("utf-8"),
                "agent_id": int(agent_history_info["id"][i]),
                "target/agent_type": np.array([int(agent_history_info["type"][i])]).reshape(1),
                "other/agent_type": np.delete(agent_history_info["type"], i, axis=0).astype(int),
                "target/is_sdc": np.array(int(agent_history_info["is_sdc"][i])).reshape(1),
                "other/is_sdc": np.delete(agent_history_info["is_sdc"], i, axis=0).astype(int),

                "target/width": agent_history_info["width"][i].item(),
                "target/length": agent_history_info["length"][i].item(),
                "other/width": np.delete(agent_history_info["width"], i),
                "other/length": np.delete(agent_history_info["length"], i),

                "target/future/xy": current_scene_agents_coordinates_future[i][None,],
                "target/future/yaw": current_scene_agents_yaws_future[i][None,],
                "target/future/valid": agent_history_info["future/valid"][i][None,],
                "target/history/xy": current_scene_target_agent_coordinates_history,
                "target/history/v_xy": current_scene_target_agent_velocities_history,
                "target/history/yaw": current_scene_target_agent_yaws_history,
                "target/history/valid": agent_history_info["history/valid"][i][None,],

                "other/future/xy": np.delete(current_scene_agents_coordinates_future, i, axis=0),
                "other/future/yaw": np.delete(current_scene_agents_yaws_future, i, axis=0),
                "other/future/valid": np.delete(agent_history_info["future/valid"], i, axis=0),
                "other/history/xy": current_scene_other_agents_coordinates_history,
                "other/history/v_xy": current_scene_other_agents_velocities_history,
                "other/history/yaw": current_scene_other_agents_yaws_history,
                "other/history/valid": np.delete(agent_history_info["history/valid"], i, axis=0),
            }

            if self._config["noisy_heading"]:
                scene_data["yaw_original"] = current_agent_scene_yaw - np.pi / 2
            array_of_scene_data_dicts.append(scene_data)
        return array_of_scene_data_dicts
