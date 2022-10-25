import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import trange
import random


class LSTMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._lstm = nn.LSTM(batch_first=True, **config)

    def forward(self, input_tensor):
        lstm_out, hidden = self._lstm(input_tensor)

        return lstm_out, hidden


class LSTMDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config
        lstm_params = {
            "input_size": config["prediction_features"],
            "hidden_size": config["hidden_size"],
            "num_layers": config["num_layers"],
        }
        linear_params = {
            "in_features": config["hidden_size"],
            "out_features": config["prediction_features"]
        }
        self._lstm = nn.LSTM(batch_first=True, **lstm_params)
        self._linear = nn.Linear(**linear_params)

    def forward(self, x_input, hidden_states):
        lstm_out, hidden = self._lstm(x_input.unsqueeze(1), hidden_states)
        output = self._linear(lstm_out.squeeze(0))

        return output, hidden


class LSTMEncoderDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config

        self._encoder = LSTMEncoder(config["encoder"])
        self._decoder = LSTMDecoder(config["decoder"])
        self._prediction_steps = config["prediction_steps"]
        self._prediction_features = config["decoder"]["prediction_features"]

    def forward(self, data):
        input_states = data["target/history/lstm_data"]
        outputs = torch.zeros(input_states.shape[0], self._prediction_steps, self._prediction_features)

        # encoder outputs
        encoder_output, encoder_hidden = self._encoder(input_states)

        decoder_input = input_states[:, -1, :self._prediction_features]
        decoder_hidden = encoder_hidden

        # predict recursively
        for t in range(self._prediction_steps):
            decoder_output, decoder_hidden = self._decoder(decoder_input, decoder_hidden)
            outputs[:, t, :] = decoder_output.squeeze(1)
            decoder_input = decoder_output.squeeze(1)

        return outputs
