import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


def calc_feat_power(dim_input, out_count, nlay, power=2):
    if dim_input == 0:
        # just bias
        features = [dim_input, out_count, out_count]
    else:
        c = (dim_input - out_count) / (nlay ** power)
        x = np.linspace(0, nlay, nlay + 2)
        y = c * x ** power + out_count
        y[-1] = dim_input
        y[0] = out_count
        features = list(y.astype(int)[::-1])
    return features


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find("Linear") != -1:
        # get the number of the inputs
        n = m.in_features
        y = 0.1
        m.weight.data.uniform_(-y, y)
        m.bias.data.uniform_(-y, y)


def reset_weights_on_net(mod, lin_r=None, bias_r=0.1, kind="uniform"):
    """
    Reset weights, putting them with random values in ranges
    'lin_r' for linear weights and `bias_r` for the bias
    """

    classname = mod.__class__.__name__
    if isinstance(mod, ZeroLinear):
        mod.reset_parameters(bias_r)
    elif classname.find("Linear") != -1:
        if kind == "uniform":
            if lin_r is None:
                n = mod.in_features
                y = 1.0 / np.sqrt(n)
                mod.weight.data.uniform_(-y, y)
            else:
                mod.weight.data.uniform_(-1 * lin_r, lin_r)
            # low = 0.1
            if mod.bias is not None:
                mod.bias.data.uniform_(-bias_r, bias_r)
        elif kind == "xavier":
            gain = nn.init.calculate_gain("relu")
            nn.init.xavier_uniform_(mod.weight.data, gain=gain)
            if mod.bias is not None:
                mod.bias.data.uniform_(-bias_r, bias_r)

    elif classname.find("Embedding") != -1:
        nn.init.normal_(mod.weight, 0.0, 1.0)


class ZeroLinear(nn.Module):
    def __init__(self, out_features, bias):
        super(ZeroLinear, self).__init__()
        self.out_features = out_features
        self.in_features = 0
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.has_bias = True
        else:
            self.bias = torch.zeros(out_features)
            self.has_bias = False
        self.reset_parameters()

    def reset_parameters(self, rang=0.1):
        if self.has_bias:
            # if kind=="uniform":
            init.uniform_(self.bias, -rang, rang)
            # elif kind=="xavier":
            #    nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        return self.bias.repeat(x.shape[0], 1)

    def extra_repr(self):
        return "in_features=0, out_features={}, bias={}".format(
            self.out_features, self.bias is not None
        )


def my_linear(in_feat, out_feat, bias):
    if in_feat > 0:
        return nn.Linear(in_feat, out_feat, bias)
    else:
        return ZeroLinear(out_feat, bias)


def make_lin_layers(features, in_func, last_func, bias):
    layers = []
    for feat_i, feat in enumerate(features[:-1]):
        """
        Code analysis: features is a list of 2 elements,
        therefore feat_i = 0 only, so far, and in_feat=features[0]
        Also, feat_i + 1 = 1, and even more the accessed
        element is always equal to q (=1 in the SI)

        This may change in the future
        """
        in_feat = feat
        out_feat = features[feat_i + 1]
        layers.append(my_linear(in_feat, out_feat, bias))
        # layers[-1].apply(weights_init_uniform_rule)

        layers.append(in_func)
    layers[-1] = last_func
    if last_func == None:
        layers.pop()
    return layers


class deep_linear(nn.Module):
    """
    Container class for layers
    """

    def __init__(
        self,
        features,
        bias,
        in_func=nn.ReLU(),
        last_func=nn.Sigmoid(),
        layer_norm=False,
    ):
        super(deep_linear, self).__init__()
        layers = []
        if features[-1] == 0:
            # this is an empty net
            return
        for feat_i, feat in enumerate(features[:-1]):

            in_feat = feat
            out_feat = features[feat_i + 1]
            mbias = True if (in_feat == 0 and out_feat > 0) else bias
            layers.append(my_linear(in_feat, out_feat, mbias))
            if layer_norm and in_feat > 0 and feat_i < len(features) - 2:
                layers.append(nn.LayerNorm([out_feat]))
            # layers[-1].apply(weights_init_uniform_rule)

            layers.append(in_func)
        layers[-1] = last_func
        if last_func == None:
            layers.pop()
        if last_func is None:
            print(layers)

        self.net = nn.Sequential(*layers)

        ##init
        if "ReLU" in repr(in_func):
            try:
                slope = in_func.negative_slope
                init_gain = nn.init.calculate_gain("leaky_relu", slope)
            except AttributeError:
                # we have a pure ReLU
                init_gain = nn.init.calculate_gain("relu")
        elif "Sigmoid" in repr(in_func):
            init_gain = nn.init.calculate_gain("sigmoid")
        else:
            init_gain = 1

    def forward(self, x):
        return self.net(x)

    def reset(self, range_weight=None, range_bias=0.1):
        fun = lambda m: reset_weights_on_net(m, range_weight, range_bias)
        self.net.apply(fun)


class MaskedDeepLinear(deep_linear):
    """
    Masked layer, that is a deep_linear when there is more
    than one value to predict (probabilities in output),
    otherwise always output 1

    dim_input: int, dimensionality of the input
    hidden_feat: list, multiplier for the intermediate layers
    """

    def __init__(
        self,
        dim_input,
        hidden_feat,
        mask,
        bias,
        in_func=nn.ReLU(),
        last_func=nn.Sigmoid(),
        scale_power=2.0,
        layer_norm=False,
    ):
        """
        mask is a 1D tensor
        """
        try:
            self.index_out = torch.where(mask)[0]
        except TypeError:
            self.index_out = torch.where(torch.tensor(mask))[0]
        self.true_out = len(mask)
        out_count = len(self.index_out)
        if out_count == 0:
            raise ValueError("Zero output")
        elif out_count == 1:
            # Do not need to calculate probabilities
            self.no_out = True
            self.out_count = 0
            features = [0]

        else:
            self.no_out = False
            self.out_count = out_count

            feat_inp = np.array(hidden_feat)
            if np.all(feat_inp < 0) or dim_input == 0:
                n_lay_want = len(hidden_feat)
                features = calc_feat_power(
                    dim_input, out_count, n_lay_want, power=scale_power
                )
            else:
                features = (
                    [dim_input]
                    + (feat_inp * dim_input).astype(int).tolist()
                    + [out_count]
                )

        self.features = tuple(features)
        super().__init__(features, bias, in_func, last_func, layer_norm=layer_norm)

    def forward(self, x):
        """
        Forward method without initializing sample index
        """
        if self.no_out:
            return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        else:
            return self.net(x)

    def reset(self, range_weight=None, range_bias=0.1):
        if not self.no_out:
            super().reset(range_weight, range_bias)

    def parameters(self, recurse: bool = True):
        if self.no_out:
            return []
        else:
            return super().parameters(recurse=recurse)

    def extra_repr(self):
        if not self.no_out:
            return (
                "first_v={}, last_v={}, ".format(self.index_out[0], self.index_out[-1])
                + super().extra_repr()
            )

        return "fixed_v={}".format(self.index_out[0])


class EmbedMaskDeepLinear(nn.Module):
    """
    Masked layer, that is a deep_linear when there is more
    than one value to predict (probabilities in output),
    otherwise always output 1

    with embedding in input
    """

    def __init__(
        self,
        inputs_neighs,
        hidden_feat,
        mask,
        bias=True,
        in_func=nn.ReLU(),
        last_func=nn.Sigmoid(),
    ):
        """
        mask is a 1D tensor
        """
        super().__init__()

        try:
            self.index_out = torch.where(mask)[0]
        except TypeError:
            self.index_out = torch.where(torch.tensor(mask))[0]
        self.true_out = len(mask)
        self.dim_input = sum(inputs_neighs)

        out_count = len(self.index_out)
        if out_count == 0:
            raise ValueError("Zero output")
        elif out_count == 1:
            # Do not need to calculate probabilities
            self.no_out = True
            self.out_count = 0
            feat = [0]
            self.features = (0, 1)

        else:
            self.no_out = False
            self.out_count = out_count

            feat = self.make_layer(inputs_neighs, hidden_feat, in_func, last_func, bias)

            self.features = tuple(feat)

    def make_layer(self, inputs_neighs, hidden_feat, in_func, last_func, bias):
        """
        Create the layer when the output is > 0
        """
        dim_input = sum(inputs_neighs)
        out_count = self.out_count
        if dim_input == 0:
            hidden_layers = [int(v * out_count) for v in hidden_feat]
            features_lin = [0] + hidden_layers + [out_count]
        else:
            base_dim = dim_input
            # the first layer is the number of outputs
            hidden_layers_1 = [int(v * base_dim) for v in hidden_feat]
            features_lin = hidden_layers_1 + [out_count]
            self.embeds = nn.ModuleList(
                [
                    nn.Embedding(neigh_out, hidden_feat[0] * base_dim)
                    for neigh_out in inputs_neighs
                ]
            )
            self.mid_lay = in_func

        layers = make_lin_layers(features_lin, in_func, last_func, bias)
        self.net = nn.Sequential(*layers)
        return features_lin if dim_input == 0 else (len(inputs_neighs), *features_lin)

    def forward(self, x):
        """
        Forward method
        """
        if self.no_out:
            return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        elif self.dim_input == 0:
            return self.net(x)
        else:

            outv = sum([self.embeds[i](x[:, i]) for i in range(len(self.embeds))])

            return self.net(self.mid_lay(outv))

    def reset(self, range_weight=None, range_bias=0.1):
        if not self.no_out:
            fun = lambda m: reset_weights_on_net(m, range_weight, range_bias)
            self.net.apply(fun)
            if self.dim_input > 0:
                self.embeds.apply(fun)

    def extra_repr(self):
        if not self.no_out:
            return (
                "first_v={}, last_v={}, ".format(self.index_out[0], self.index_out[-1])
                + super().extra_repr()
            )

        return "fixed_v={}".format(self.index_out[0])


class TwoNetCascade(nn.Module):
    """
    Network used to get the results in cascade:
    x -> y1
    x,y1 -> y2
    """

    def __init__(self, features, bias, in_func=nn.ReLU(), last_func=nn.Sigmoid()):
        super(TwoNetCascade, self).__init__()
        self.feat_first = list(features)
        self.first_net = deep_linear(
            features, bias, in_func=in_func, last_func=last_func
        )

        self.feat_second = list(features)
        self.feat_second[0] += self.feat_first[-1]

        self.second_net = deep_linear(
            self.feat_second, bias, in_func=in_func, last_func=last_func
        )
        self.device = "cpu"

    def forward(self, x):
        first_res = self.first_net(x)
        second_res = self.second_net(torch.cat((x, first_res), dim=1))

        return first_res, second_res

    def to(self, device):
        self.first_net.to(device)
        self.second_net.to(device)
        self.device = device
