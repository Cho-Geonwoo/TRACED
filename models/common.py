# Copyright (c) 2017 Ilya Kostrikov
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of
# https://github.com/facebookresearch/dcd/blob/main/models/common.py
# which is a modified version of
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This file has been modified for the TRACED
# Note: Author information anonymized for double-blind review.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rnn import LSTM


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

init_relu_ = lambda m: init(
    m,
    nn.init.orthogonal_,
    lambda x: nn.init.constant_(x, 0),
    nn.init.calculate_gain("relu"),
)

init_tanh_ = lambda m: init(
    m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
)


def apply_init_(modules, gain=None):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if gain:
                nn.init.xavier_uniform_(m.weight, gain=gain)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def weight_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.LSTM):
        for name, p in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0.0)
                # forget‑gate bias = 1
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class DeviceAwareModule(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "same")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "valid":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class RNN(nn.Module):
    """
    Actor-Critic network (base class)
    """

    def __init__(self, input_size, hidden_size=128, arch="lstm"):
        super().__init__()

        self.arch = arch
        self.is_lstm = arch == "lstm"

        self._hidden_size = hidden_size
        if arch == "gru":
            self.rnn = nn.GRU(input_size, hidden_size)
        elif arch == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported RNN architecture {arch}.")

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x, hxs, masks):
        if self.is_lstm:
            # Since nn.LSTM defaults to all zero states if passed None state
            hidden_batch_size = x.size(0) if hxs is None else hxs[0].size(0)
        else:
            hidden_batch_size = hxs.size(0)

        if x.size(0) == hidden_batch_size:
            masked_hxs = (
                tuple((h * masks).unsqueeze(0) for h in hxs)
                if self.is_lstm
                else (hxs * masks).unsqueeze(0)
            )

            x, hxs = self.rnn(x.unsqueeze(0), masked_hxs)
            x = x.squeeze(0)

            hxs = tuple(h.squeeze(0) for h in hxs) if self.is_lstm else hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs[0].size(0) if self.is_lstm else hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = (h.unsqueeze(0) for h in hxs) if self.is_lstm else hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                masked_hxs = (
                    tuple(h * masks[start_idx].view(1, -1, 1) for h in hxs)
                    if self.is_lstm
                    else hxs * masks[start_idx].view(1, -1, 1)
                )
                rnn_scores, hxs = self.rnn(x[start_idx:end_idx], masked_hxs)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = tuple(h.squeeze(0) for h in hxs) if self.is_lstm else hxs.squeeze(0)

        return x, hxs


def one_hot(dim, inputs, device="cpu"):
    one_hot = torch.nn.functional.one_hot(inputs.long(), dim).squeeze(1).float()
    return one_hot


def make_fc_layers_with_hidden_sizes(sizes, input_size):
    fc_layers = []
    for i, layer_size in enumerate(sizes[:-1]):
        input_size = input_size if i == 0 else sizes[0]
        output_size = sizes[i + 1]
        fc_layers.append(init_tanh_(nn.Linear(input_size, output_size)))
        fc_layers.append(nn.Tanh())

    return nn.Sequential(*fc_layers)


class ImgEncoder(nn.Module):
    """(B,3,5,5)  →  (B, z_dim)"""

    def __init__(self, z_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),  # 32×5×5
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1),  # 64×5×5
            nn.ReLU(inplace=True),
            nn.Flatten(),  # 64*5*5 = 1600
        )
        self.lin = nn.Linear(1600, z_dim)

    def forward(self, x):  # x : (B,C=3,5,5)
        h = self.cnn(x)
        return self.lin(h)  # (B,z_dim)


class ImgDecoder(nn.Module):
    """(B, z_dim)  →  (B,3,5,5)"""

    def __init__(self, z_dim=128, out_shape=(3, 5, 5)):
        super().__init__()
        C, H, W = out_shape  # 3,5,5
        self.lin = nn.Linear(z_dim, 64 * 5 * 5)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 1, 1),  # 32×5×5
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, C, 3, 1, 1),  # 3×5×5
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.lin(z).view(-1, 64, 5, 5)
        return self.deconv(x)


class ImageTransitionPredictionModel(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        enc_dim=128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.img_encoder = ImgEncoder(z_dim=enc_dim)
        self.img_decoder = ImgDecoder(z_dim=hidden_dim, out_shape=state_dim)

        self.lstm = LSTM(
            input_size=enc_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, img, action, hidden=None):
        img = self.img_encoder(img)
        x = torch.cat([img, action], dim=-1)
        x, hidden = self.lstm(x, hidden)
        x = self.img_decoder(x)
        return x, hidden


class FlatTransitionPredictionModel(nn.Module):
    def __init__(
        self, state_dim, action_dim, hidden_dim=256, num_layers=1, dropout=0.0
    ):
        super().__init__()
        self.lstm = LSTM(
            input_size=state_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action, hidden=None):
        x = torch.cat([state, action], dim=-1)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden
