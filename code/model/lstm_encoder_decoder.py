import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from tqdm import trange
import random


class LSTMDecoderTorchOld(nn.Module):
    def __init__(self, predicted_features, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=predicted_features, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, predicted_features)

    def forward(self, x_input, hidden_states):
        '''
        : param x_input:
        : param hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''

        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(1), hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class LSTMEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._lstm = nn.LSTM(batch_first=True, **config)

    def forward(self, input_tensor):
        # lstm_out, hidden = self._lstm(
        #     input_tensor.view(input_tensor.shape[0], input_tensor.shape[1], self.observed_features))

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


class PyTorchModel:

    def __init__(self, trainer_config, load=False, use_GPU=True, **kwargs):
        self.ONLY_TTP = trainer_config["only_ttp"]
        self.DOWNSAMPLER = trainer_config["downsampler"]
        self.TIME_DEPENDENT_WEIGHTS = trainer_config["time_dependent_weights"]
        self.SAVE_FOLDER = trainer_config["save_folder"]
        self.MODEL_NAME = trainer_config["model_name"]
        self.MAX_EPOCHS = trainer_config["max_epochs"]
        self.PATIENCE = trainer_config["patience"]
        self.use_GPU = use_GPU
        self.epoch = 0
        self.checkpoint_path = f"{self.SAVE_FOLDER}checkpoints/{self.MODEL_NAME}.tar"
        self.save_path = f"{self.SAVE_FOLDER}{self.MODEL_NAME}.tar"

    def summary(self):
        print("Model summary")
        print(self)

    def _as_torch(self, tensor):
        # converts input TF tensor to Torch tensors.
        # ToDo: optimize in data loader
        # return torch.from_numpy(tensor)
        return torch.from_numpy(tensor.numpy())

    def _batch_as_torch(self, batch):
        batch_torch = {}
        for key in batch.keys():
            if batch[key].dtype == tf.dtypes.string:  # ignore for now, because pytorch doesn't like strings
                batch_torch[key] = batch[key]
            else:
                batch_torch[key] = self._as_torch(batch[key])
                batch_torch[key] = self._move_to_gpu(batch_torch[key])
        return batch_torch

    def _move_to_gpu(self, inp):
        if self.use_GPU and torch.cuda.is_available():
            try:
                inp = inp.cuda()
            except RuntimeError:
                print("Failed to allocate to GPU")
                pass  # likely out of memory in GPU
        return inp

    def _init_losses(self, n_epochs):
        if self.epoch == 0:  # training from scratch. Fill everything with nan
            self.losses_train = np.full(n_epochs, np.nan)
            self.losses_val = np.full(n_epochs, np.nan)

        else:  # check for shape mismatches to prevent errors later on
            print("Resuming training at epoch ", self.epoch)
            losses_length = len(self.losses_train)
            if n_epochs > losses_length:
                print("Resizing losses vector to prevent explosions")
                new_losses_train = np.full(n_epochs, np.nan)
                new_losses_val = np.full(n_epochs, np.nan)

                new_losses_train[:losses_length] = np.copy(self.losses_train)
                new_losses_val[:losses_length] = np.copy(self.losses_val)

                self.losses_train = new_losses_train
                self.losses_val = new_losses_val

                # make bigger loss vector, copy everything from current loss vector
            elif n_epochs < losses_length:
                # no big deal, the lossess will have some nan values at the end
                # ToDo: resize to avoid nan
                pass
            # else: no problems


class LSTMAutoencoderTorch(nn.Module, PyTorchModel):
    """
    LSTM Autoencoder. Base code taken from https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py
    and adapted for our purposes

    Note: at the moment, if multiple lstms are stacked for the encoder and decoder, it is assume the have the same
    hidden state size
    """

    def __init__(self, trainer_config, load=False, use_GPU=True, **kwargs):
        nn.Module.__init__(self)
        PyTorchModel.__init__(self, trainer_config, load=load, use_GPU=use_GPU, **kwargs)

        # self.observed_features = trainer_config["observed_features"]-1
        self.observed_features = trainer_config["observed_features"]

        self.architecture_encoder = kwargs["architecture_encoder"]
        self.architecture_decoder = kwargs["architecture_decoder"]

        num_layers_encoder = len(self.architecture_encoder)
        num_layers_decoder = len(self.architecture_decoder)

        self.encoder = LSTMEncoderTorch(observed_features=self.observed_features,
                                        hidden_size=self.architecture_encoder[0],
                                        num_layers=num_layers_encoder)
        self.decoder = LSTMDecoderTorch(predicted_features=2,
                                        hidden_size=self.architecture_decoder[0],
                                        num_layers=num_layers_decoder)

        # self.loss_fn = WeightedL1Loss()  # ToDo: customize depending on trainer config
        self.loss_fn = WeightedLoss(loss_fn=F.huber_loss)  # ToDo: customize depending on trainer config

        self.optimizer = optim.Adam(self.parameters(),
                                    lr=trainer_config["lr"])  # ToDo: customize depending on trainer config

        if load:
            self._load()

        if use_GPU and torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

    def _get_sample_weights_linear(self, weights):
        num_samples, num_timesteps = weights.shape[0], weights.shape[1]
        sample_weights = np.linspace(1, 0.1, num_timesteps).reshape(1, num_timesteps)
        sample_weights = sample_weights / sample_weights.sum()
        sample_weights = torch.from_numpy(sample_weights)

        if self.use_GPU and torch.cuda.is_available():
            sample_weights = sample_weights.cuda()
            weights = weights.cuda()

        weights = torch.mul(weights, sample_weights)
        return weights

    def _get_training_targets(self, inputs):
        sample_is_valid, states_is_valid = inputs['sample_is_valid'], inputs['states_is_valid']
        if "states_normalized" in inputs:
            states = inputs['states_normalized']
        elif 'states_modified' in inputs:
            states = inputs['states_modified']
        else:
            states = inputs['states']

        if self.ONLY_TTP:
            # sample_is_valid_and_ttp = tf.math.logical_and(sample_is_valid, inputs['tracks_to_predict'])
            sample_is_valid_and_ttp = torch.logical_and(sample_is_valid, inputs['tracks_to_predict'])
            sample_is_valid = sample_is_valid_and_ttp

        states_observed, states_future = states[:, :, :11, :], states[:, :, 11:, :]
        states_future_is_valid = states_is_valid[:, :, 11:]

        states_observed = states_observed[sample_is_valid]
        states_future = states_future[sample_is_valid]
        states_future_is_valid = states_future_is_valid[sample_is_valid]

        # Set training target.
        gt_targets = states_future[:, ::self.DOWNSAMPLER, :2]
        weights = states_future_is_valid[:, ::self.DOWNSAMPLER]

        if self.TIME_DEPENDENT_WEIGHTS:
            weights = self._get_sample_weights_linear(weights)

        return states_observed, gt_targets, weights

    def train_step(self, input_states, target_states, prediction_steps, training_prediction, teacher_forcing_ratio):
        outputs = torch.zeros(input_states.shape[0], prediction_steps, 2)
        outputs = self._move_to_gpu(outputs)

        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_states)
        # encoder_output, encoder_hidden = self.encoder(input_states[..., [0,1,3]])

        # decoder with teacher forcing
        decoder_input = input_states[:, -1, :2]
        decoder_hidden = encoder_hidden

        if training_prediction == 'recursive':
            for t in range(prediction_steps):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output.squeeze(1)
                decoder_input = decoder_output.squeeze(1)

        if training_prediction == 'teacher_forcing':
            # use teacher forcing
            if random.random() < teacher_forcing_ratio:
                for t in range(prediction_steps):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = target_states[:, t, :]

            # predict recursively
            else:
                for t in range(prediction_steps):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output.squeeze(1)

        if training_prediction == 'mixed_teacher_forcing':
            # predict using mixed teacher forcing
            for t in range(prediction_steps):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output.squeeze(1)

                # predict with teacher forcing
                if random.random() < teacher_forcing_ratio:
                    decoder_input = target_states[:, t, :]

                # predict recursively
                else:
                    decoder_input = decoder_output.squeeze(1)

        # compute the loss

        return outputs

    def train_epoch(self, data_train, prediction_steps, training_prediction, teacher_forcing_ratio, tr):
        n_batches_train = 0  # counter for number of batches

        ###### Train
        self.train()
        loss_train = 0.
        for b, batch in enumerate(data_train):

            # print(tf.batch["sample_is_valid"].any())
            if not tf.reduce_any(batch["sample_is_valid"]):
                continue

            batch = self._batch_as_torch(batch)

            input_states, target_states, weights = self._get_training_targets(batch)

            # zero the gradient
            self.optimizer.zero_grad()
            outputs = self.train_step(input_states, target_states, prediction_steps, training_prediction,
                                      teacher_forcing_ratio)
            loss = self.loss_fn(outputs, target_states, weights)
            loss_train += loss.item()

            # backpropagation
            loss.backward()
            self.optimizer.step()

            n_batches_train += 1

            tr.set_postfix(loss_train="{0:.2e}".format(loss_train / n_batches_train), batch=n_batches_train)

        # loss for epoch
        loss_train /= n_batches_train

        # print("num_batches_train", n_batches_train)

        return loss_train

    def val_epoch(self, data_val, prediction_steps):
        self.eval()
        n_batches_val = 0  # counter for number of batches
        loss_val = 0.
        for b, batch in enumerate(data_val):
            if not tf.reduce_any(batch["sample_is_valid"]):
                continue

            batch = self._batch_as_torch(batch)

            input_states, target_states, weights = self._get_training_targets(batch)
            outputs = self.predict(input_states, prediction_steps, return_np=False)
            outputs = self._move_to_gpu(outputs)

            loss = self.loss_fn(outputs, target_states, weights)

            loss_val += loss.item()

            n_batches_val += 1

        # loss for epoch
        loss_val /= n_batches_val

        return loss_val

    def train_model(self, data_train, data_val, prediction_steps,
                    strategy='recursive', teacher_forcing_ratio=0.5,
                    dynamic_tf=True):

        # initialize array of losses
        n_epochs = self.MAX_EPOCHS
        self._init_losses(n_epochs)
        print("Training strategy: ", strategy)
        epochs_without_improvement = 0

        with trange(self.epoch, n_epochs) as tr:
            for it in tr:
                loss_train = self.train_epoch(data_train, prediction_steps, strategy, teacher_forcing_ratio,
                                              tr)

                self.losses_train[it] = loss_train

                loss_val = self.val_epoch(data_val, prediction_steps)
                self.losses_val[it] = loss_val

                print("Epoch losses: ", "train: ", loss_train, "val: ", loss_val)

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                    # progress bar
                tr.set_postfix(loss_train="{0:.2e}".format(loss_train),
                               loss_val="{0:.2e}".format(loss_val))

                if loss_val == np.nanmin(self.losses_val):
                    print("Best validation loss. Saving checkpoint...")
                    self.save_checkpoint()
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.PATIENCE:
                    print("Max patience reached. Restoring best model and saving...")
                    self._load(from_checkpoint=True)
                    print("self.losses_val", self.losses_val)
                    print("self.losses_train", self.losses_train)
                    return self.losses_train, self.losses_val
                    # self.save()

                self.epoch += 1

        return self.losses_train, self.losses_val

    def predict(self, input_states, prediction_steps, prediction_features=2, return_np=True):
        outputs = torch.zeros(input_states.shape[0], prediction_steps, prediction_features)

        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_states)
        # encoder_output, encoder_hidden = self.encoder(input_states[..., [0,1,3]])

        # decoder with teacher forcing
        decoder_input = input_states[:, -1, :prediction_features]
        decoder_hidden = encoder_hidden

        # predict recursively
        for t in range(prediction_steps):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t, :] = decoder_output.squeeze(1)
            decoder_input = decoder_output.squeeze(1)

        if return_np:
            return outputs.detach().numpy()

        return outputs

    def save(self, path=None):
        if path is None:  # save to default path
            path = self.save_path
        if ".tar" not in path:
            path += ".tar"
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn': self.loss_fn,
            'losses_train': self.losses_train,
            'losses_val': self.losses_val,
        }, path)

    def save_checkpoint(self):
        self.save(self.checkpoint_path)

    def _load(self, from_checkpoint=False):

        if self.use_GPU and torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()

        path = self.checkpoint_path if from_checkpoint else self.save_path
        print("Loading model from: ", path)

        if self.use_GPU and torch.cuda.is_available():
            checkpoint = torch.load(path)
        else:
            checkpoint = torch.load(path, map_location=torch.device('cpu'))

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss_fn = checkpoint['loss_fn']  # currently overrides
        self.losses_train = checkpoint['losses_train']
        self.losses_val = checkpoint['losses_val']
