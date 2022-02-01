import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..net import base as common_net
from ..net import deep_linear
from ..utils import conf_utils as c_utils


def make_masks(observ, n, T):
    """
    Create masks from the observations list
    #order: time, node, value
    order: node_i, state/value, time
    """
    masks = np.full((n, 2, T + 1), True)
    times = np.arange(T + 1)
    for line in observ:
        # print(i,"\n",r)
        mask_inf = np.zeros_like(times, dtype=np.bool)  # masks[r.node,0]
        mask_rec = np.zeros_like(times, dtype=np.bool)  # masks[r.node,1]
        t_obs = line[2]
        idx_obs = line[0]
        val_obs = line[1]
        if val_obs == 0:  # susc
            mask_inf[times > t_obs] = True
            mask_rec[times > t_obs] = True
        elif val_obs == 1:  # inf
            mask_inf[times <= t_obs] = True
            mask_rec[times > t_obs] = True
        elif val_obs == 2:  ##rec
            mask_inf[times <= t_obs] = True
            mask_rec[times <= t_obs] = True

        else:
            raise ValueError("Invalid observation value")
        # print(i, masks[r.node,0].view(np.int8),  masks[r.node,1].view(np.int8))
        if np.all(mask_inf == False):
            mask_inf[-1] = True
        if np.all(mask_rec == False):
            mask_rec[-1] = True
        masks[idx_obs, 0] &= mask_inf
        masks[idx_obs, 1] &= mask_rec
    if len(observ) > 0:
        # print(masks[idx_obs,0].view(np.int8))
        # print(masks[idx_obs,1].view(np.int8))
        pass
    return masks


class SIRPathColdObs(common_net.Autoreg):
    """
    Compact SI Path
    samples stored non-one-hot
    """

    q = 3

    def __init__(
        self,
        neighs,
        T: int,
        obs_list: list,
        hidden_layer_spec: list,
        bias: bool = True,
        min_value_prob=1e-40,
        in_func=nn.ReLU(),
        last_func=nn.Softmax(dim=1),
        device="cpu",
        dtype=torch.float,
        lin_scale_power: float = 2.0,
        layer_norm: bool = False,
    ):
        """
        Observations have to be in a list of values
        (node_i, state/value as integer, time of obs)
        """
        super().__init__(device, dtype)

        self.N = len(neighs)
        self.T = T
        self.num_feat = T + 1
        if isinstance(neighs, dict):
            self.true_neighs = []
            self.nodes_labels = []
            for k, v in neighs.items():
                self.true_neighs.append(sorted(v))
                self.nodes_labels.append(k)
        else:
            vals = neighs
            self.true_neighs = [sorted(x) for x in neighs]
            self.nodes_labels = list(range(len(neighs)))

        self.num_nets = self.N
        self.data_basic_shape = (self.N, 2)
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.min_value_prob = min_value_prob
        self.linear_net_scaling = lin_scale_power
        self.layer_norm = layer_norm

        self.masks = torch.tensor(make_masks(obs_list, self.N, self.T))
        # for i in range(self.N):
        #    print(self.masks[i].to(torch.int8))

        self.sublayers = []
        n_digits = int(np.ceil(np.log10(self.num_nets)))
        # Build sublayers
        for n_i in range(self.num_nets):
            layer = self.build_layer(
                n_i, hidden_layer_spec, bias, in_func, last_func, lin_scale_power
            )
            layer.to(device=device)
            self.sublayers.append(layer)
            self.add_module("lay_{n:0{width}d}".format(n=n_i, width=n_digits), layer)

        self.params = []
        for i in range(self.num_nets):
            pars = filter(lambda p: p.requires_grad, self.sublayers[i].parameters())
            self.params.extend(pars)
        # self.params = list(filter(lambda p: p.requires_grad, self.params))
        # self.params = list(self.params)
        self.nparams = int(sum([np.prod(p.shape) for p in self.params]))

        self.sample_dtype = torch.long
        """
        self.sample_neighs =[]
        for i,nes in enumerate(self.true_neighs):
            if self.sublayers[i][0].no_out and self.sublayers[i][1].no_out:
                self.sample_neighs.append(torch.tensor([]).long().to(device))
            else:
                self.sample_neighs.append(
                    torch.tensor(
                        list(filter(lambda x: not (self.sublayers[x][0].no_out), nes))
                        ).long().to(device)
                )
        print("{} active nets".format(len(self.dimensions(True))))
        """
        self.sample_neighs = [torch.tensor(ne) for ne in self.true_neighs]
        masks_s = self.masks.detach().clone()
        masks_s[torch.where(masks_s.sum(-1) <= 1)] = False
        self.masks_sample = masks_s.to(self.device)
        self.bselect = None

        self.params_i = {}

        for i in range(self.num_nets):
            self.params_i[i] = tuple(
                filter(lambda p: p.requires_grad, self.sublayers[i].parameters())
            )

        # print(self.masks.to(int))
        # for n in self.sublayers:
        #    print((n[0].out_count, n[1].out_count))
        # print(self.dimensions())

    def build_layer(self, idx, layer_spec, bias, in_func, last_func, scale_power):
        """
        Build new linear layer
        """
        n_input = 0
        for neig in self.true_neighs[idx]:
            n_input += self.sublayers[int(neig)][0].out_count
            n_input += self.sublayers[int(neig)][1].out_count

        kwargs = dict(scale_power=scale_power, layer_norm=self.layer_norm)
        net_inf = deep_linear.MaskedDeepLinear(
            n_input, layer_spec, self.masks[idx][0], bias, in_func, last_func, **kwargs
        )
        n_input_rec = n_input + net_inf.out_count
        net_rec = deep_linear.MaskedDeepLinear(
            n_input_rec,
            layer_spec,
            self.masks[idx][1],
            bias,
            in_func,
            last_func,
            **kwargs
        )

        return nn.ModuleList((net_inf, net_rec))

    def init(self, method="uniform", lin_r=None, bias_r=0.1):
        for lay in self.sublayers:
            for mskdeep in lay:
                if mskdeep.out_count == 0:
                    continue
                for mod in mskdeep.net:
                    deep_linear.reset_weights_on_net(
                        mod, lin_r=lin_r, bias_r=bias_r, kind=method
                    )

    def dimensions(self, active_only=False):
        return [[nett.features for nett in lay] for lay in self.sublayers]

    def extract_idx_samples(self, samples_cold, net_i):
        """
        Convenience function to properly extract the samples
        and the outgoing nodes indices for a net
        input the samples (non one hot) and the subnet index
        """
        neighs_i = self.sample_neighs[net_i]
        indix = self.nodes_labels[net_i]
        n_samples = samples_cold.shape[0]
        avoid_samples = (self.sublayers[net_i][0].out_count == 0) and (
            self.sublayers[net_i][1].out_count == 0
        )
        if avoid_samples:
            samples_select = torch.zeros((n_samples, 0), device=self.device)
        elif len(neighs_i) > 0:
            # get the times of the neighbors
            times = samples_cold[:, neighs_i]
            # select the masks of the neighbors and flatten them
            masks_sel = self.masks_sample[neighs_i].view(-1)
            # put the samples on onehot and apply masks
            samples_hot = F.one_hot(times, self.num_feat).view(n_samples, -1)
            samples_select = samples_hot[:, masks_sel].to(self.dtype)
            # print(samples_select.shape)
            del samples_hot, times, masks_sel
        else:
            # the input vector should have dimension zero
            samples_select = torch.zeros((n_samples, 0), device=self.device)
        return indix, samples_select

    def _log_prob(self, samples, probs):
        ## TODO: check
        # print(samples.shape, self.data_basic_shape)
        assert samples.shape[1:] == self.data_basic_shape
        log_prob = torch.log(probs.clamp(min=self.min_value_prob))
        # print(log_prob.shape)
        return log_prob.sum(-1).sum(-1)

    def _get_trec_probs(self, i, samples_inf, inf_idx_times, batch_size):
        n_out_inf = self.sublayers[i][0].out_count
        if n_out_inf > 0:
            t_inf_hot = F.one_hot(inf_idx_times, n_out_inf).view(batch_size, -1)

            t_rec_probs = self.sublayers[i][1](
                torch.cat((samples_inf, t_inf_hot.to(self.dtype)), dim=-1)
            )
        else:
            ### the infection time is fixed
            t_rec_probs = self.sublayers[i][1](samples_inf)
        return t_rec_probs

    def sample(self, batch_size):
        """
        Sample the network, computing probabilities first and
        them sum them
        """
        data_shape = (batch_size, self.N, 2)
        samples = self.get_empty_matrix(data_shape, data_type=self.sample_dtype)
        samples.requires_grad_(False)
        # samples_hot = torch.zeros(num_s,N,2,T+1,device=device)

        probs = self.get_empty_matrix(data_shape)
        bselect = torch.arange(batch_size, dtype=torch.long, device=self.device)
        # print(samples_hot)
        for i in range(self.num_nets):
            # print(f"Node {i} inf",end="\r")
            with torch.no_grad():
                indix, samples_select = self.extract_idx_samples(samples, i)
            # print(i,indix,samples_select)
            # Infection times
            if self.sublayers[i][0].out_count == 0:
                ##avoid random draw
                with torch.no_grad():
                    idx_times = None
                    samples[:, indix, 0] = self.sublayers[i][0].index_out[0]
                probs[:, indix, 0] = 1.0
            else:
                t_inf_probs = self.sublayers[i][0](samples_select)
                with torch.no_grad():
                    ## sample tinf
                    idx_times = torch.multinomial(t_inf_probs, 1).squeeze()  # .detach()
                    samples[:, indix, 0] = idx_times + self.sublayers[i][0].index_out[0]
                probs[:, indix, 0] = t_inf_probs[bselect, idx_times]
                del t_inf_probs
            # print(f"Node {i} rec", end="\r")
            t_rec_probs = self._get_trec_probs(i, samples_select, idx_times, batch_size)
            ### don't do random draws when we are certain of the time
            if self.sublayers[i][1].out_count == 0:
                with torch.no_grad():
                    idx_times = torch.ones(
                        (batch_size, 1), device=self.device, dtype=samples.dtype
                    )
                    samples[:, indix, 1] = self.sublayers[i][1].index_out[0]
                probs[:, indix, 1] = 1.0
            else:
                with torch.no_grad():
                    ## sample tinf
                    idx_trec = torch.multinomial(t_rec_probs, 1).squeeze()  # .detach()
                    samples[:, indix, 1] = idx_trec + self.sublayers[i][1].index_out[0]
                probs[:, indix, 1] = t_rec_probs[bselect, idx_trec]
                del idx_trec

            del indix, samples_select
            # samples[bselect, indix, times] = 1

        return samples.detach(), probs

    def forward(self, x):
        """
        Compute the probabilities
        """
        batch_size = x.shape[0]
        shape = (batch_size, *self.data_basic_shape)
        bselect = torch.arange(batch_size, dtype=torch.long, device=self.device)
        probs = self.get_empty_matrix(shape)
        for i in range(self.num_nets):
            indix, samples_select = self.extract_idx_samples(x, i)
            t_inf_probs = self.sublayers[i][0](samples_select)
            sam_indics = x[:, indix, 0] - self.sublayers[i][0].index_out[0]
            probs[:, indix, 0] = t_inf_probs[bselect, sam_indics]

            t_rec_probs = self._get_trec_probs(
                i, samples_select, sam_indics, batch_size
            )

            sam_indics = x[:, indix, 1] - self.sublayers[i][1].index_out[0]
            probs[:, indix, 1] = t_rec_probs[bselect, sam_indics]

        return probs

    def log_prob_i(self, node_i, samples):
        """
        Compute log prob of just one node
        """
        batch_size = samples.shape[0]
        bselect = torch.arange(
            batch_size, dtype=torch.long, device=self.device, requires_grad=False
        )
        # get relevant samples
        indix, samples_select = self.extract_idx_samples(samples, node_i)
        t_inf_probs = self.sublayers[node_i][0](samples_select)
        sam_indics = samples[:, indix, 0] - self.sublayers[node_i][0].index_out[0]
        probs_i_sel = t_inf_probs[bselect, sam_indics]

        t_rec_probs = self._get_trec_probs(
            node_i, samples_select, sam_indics, batch_size
        )

        sam_indics = samples[:, indix, 1] - self.sublayers[node_i][1].index_out[0]
        probs_r_sel = t_rec_probs[bselect, sam_indics]

        log_prob_i = torch.log(probs_i_sel.clamp(min=self.min_value_prob)) + torch.log(
            probs_r_sel.clamp(min=self.min_value_prob)
        )
        assert len(log_prob_i.shape) == 1 and log_prob_i.shape[0] == batch_size
        return log_prob_i

    def transform_samples(self, samples):
        """
        Transform samples to make them ready for SIR energy calculation

        This is valid for the non 1-hot energy calculation
        TODO: Check
        """
        assert samples.shape[1:] == self.data_basic_shape
        return samples

    def marginals_(self, samples, batch_size=1000):
        """
        Compute marginals transforming in one hot
        One batch at a time
        TODO: Check
        """
        view_batch = torch.split(samples, batch_size)
        num_samples = samples.shape[0]
        marginals = self.get_empty_matrix((self.N, self.T, self.q))
        for mini_batch in view_batch:
            x_hot = c_utils.one_hot_conf_from_times(mini_batch, self.T)

            marginals += x_hot.sum(dim=0)
        return marginals / num_samples

    def marginals(self, num_samples=10000, batch_size=100):
        with torch.no_grad():
            samples, prob = self.sample(num_samples)
        return self.marginals_(samples, batch_size=batch_size)
