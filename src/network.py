from typing import List
from copy import deepcopy

import torch
import torch.nn as nn

from catalyst import utils


def _get_convolution_net(
    in_channels: int,
    history_len: int = 1,
    channels: List = None,
    kernel_sizes: List = None,
    strides: List = None,
    use_bias: bool = False,
    use_groups: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    channels = channels or [32, 64, 32]
    kernel_sizes = kernel_sizes or [8, 4, 3]
    strides = strides or [4, 2, 1]
    activation_fn = torch.nn.__dict__[activation]
    assert len(channels) == len(kernel_sizes) == len(strides)

    def _get_block(**conv_params):
        layers = [nn.Conv2d(**conv_params)]
        if use_normalization:
            layers.append(nn.InstanceNorm2d(conv_params["out_channels"]))
        if use_dropout:
            layers.append(nn.Dropout2d(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    channels.insert(0, history_len * in_channels)
    params = []
    for i, (in_channels, out_channels) in enumerate(utils.pairwise(channels)):
        num_groups = 1
        if use_groups:
            num_groups = history_len if i == 0 else 4
        params.append(
            {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "bias": use_bias,
                "kernel_size": kernel_sizes[i],
                "stride": strides[i],
                "groups": num_groups,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    # input_shape: tuple = (3, 84, 84)
    # conv_input = torch.Tensor(torch.randn((1,) + input_shape))
    # conv_output = net(conv_input)
    # torch.Size([1, 32, 7, 7]), 1568
    # print(conv_output.shape, conv_output.nelement())

    return net


def _get_linear_net(
    in_features: int,
    history_len: int = 1,
    features: List = None,
    use_bias: bool = False,
    use_normalization: bool = False,
    use_dropout: bool = False,
    activation: str = "ReLU"
) -> nn.Module:

    features = features or [64, 128, 64]
    activation_fn = torch.nn.__dict__[activation]

    def _get_block(**linear_params):
        layers = [nn.Linear(**linear_params)]
        if use_normalization:
            layers.append(nn.LayerNorm(linear_params["out_features"]))
        if use_dropout:
            layers.append(nn.Dropout(p=0.1))
        layers.append(activation_fn(inplace=True))
        return layers

    features.insert(0, history_len * in_features)
    params = []
    for i, (in_features, out_features) in enumerate(utils.pairwise(features)):
        params.append(
            {
                "in_features": in_features,
                "out_features": out_features,
                "bias": use_bias,
            }
        )

    layers = []
    for block_params in params:
        layers.extend(_get_block(**block_params))

    net = nn.Sequential(*layers)
    net.apply(utils.create_optimal_inner_init(activation_fn))

    return net


class StateNet(nn.Module):
    def __init__(
        self,
        main_net: nn.Module,
        observation_net: nn.Module = None,
        aggregation_net: nn.Module = None,
    ):
        """
        Abstract network, that takes some tensor
        T of shape [bs; history_len; ...]
        and outputs some representation tensor R
        of shape [bs; representation_size]

        input_T [bs; history_len; in_features]

        -> observation_net (aka observation_encoder) ->

        observations_representations [bs; history_len; obs_features]

        -> aggregation_net (flatten in simplified case) ->

        aggregated_representation [bs; hid_features]

        -> main_net ->

        output_T [bs; representation_size]

        Args:
            main_net:
            observation_net:
            aggregation_net:
        """
        super().__init__()
        self.main_net = main_net
        self.observation_net = observation_net
        self.aggregation_net = aggregation_net

    def forward(self, state):
        x = state

        x = x / 255.
        batch_size, history_len, c, h, w = x.shape

        x = x.view(batch_size, -1, h, w)
        x = self.observation_net(x)

        x = x.view(batch_size, -1)
        x = self.main_net(x)
        return x

    @classmethod
    def get_from_params(
        cls,
        observation_net_params=None,
        # aggregation_net_params=None,
        main_net_params=None,
    ) -> "StateNet":

        observation_net_params = deepcopy(observation_net_params)
        main_net_params = deepcopy(main_net_params)

        mult_ = 7 * 7
        observation_net = _get_convolution_net(**observation_net_params)
        observation_net_out_features = \
            observation_net_params["channels"][-1] * mult_

        aggregation_net = None
        main_net_in_features = observation_net_out_features

        main_net_params["in_features"] = main_net_in_features
        main_net = _get_linear_net(**main_net_params)

        net = cls(
            observation_net=observation_net,
            aggregation_net=aggregation_net,
            main_net=main_net
        )

        return net
