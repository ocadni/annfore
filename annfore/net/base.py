import numpy as np
import torch
import torch.nn as nn

from ..learn import l_utils

print_batch_step = l_utils.print_batch_step

class Autoreg(nn.Module):

    def __init__(self, device, dtype=torch.float):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype


    def sample(self, batch_size):
        """
        Sample the network
        """
        raise NotImplementedError

    def transform_samples(self, samples):
        """
        Transform the samples for energy calculation
        """
        raise NotImplementedError

    def _log_prob(self, samples, probs):
        """
        Calculate log Q of the probability of the samples
        """
        raise NotImplementedError

    def log_prob(self, samples):
        probs = self.forward(samples) 
        return self._log_prob(samples, probs)


    def get_empty_matrix(self, dims, data_type="undef"):
        """
        Initialize empty matrix with zeros
        """
        if data_type == "undef":
            data_type = self.dtype
        return torch.zeros(dims, dtype=data_type, device=self.device)

    def get_count_parameters(self, sublayers):
        params = []
        for i, net in enumerate(sublayers):
            pars = filter(lambda p: p.requires_grad,
                          net.parameters())
            params.extend(pars)

        nparams = int(sum([np.prod(p.shape) for p in params]))
        return nparams, params

    def _extract_marginals(self, sample_size):
        """
        Extract marginals with sample size
        """
        samples_raw, _ = self.sample(sample_size)
        return self.marginals_(samples_raw)

    def marginals_(self, samples_batch):
        """
        Return marginals from samples
        Name comes from the training interface

        Put here all the trasformations needed

        Return the marginals on the samples batch
        """
        raise NotImplementedError


    def marginals(self, num_samples, max_batch=1000, verbose=False):
        """
        Compute marginals with (max) batch size `max_batch`, up to `num_samples`
        """

        if verbose:
            print("Source extraction")
        with torch.no_grad():
            num_done = 0
            iter_ = 0
            
            final_counters = None
            if verbose:
                print_batch_step(iter_, max_batch, num_samples)
            while iter_ < num_samples:
                num_done += max_batch
                #print(M_i, Z_i)
                sam = self._extract_marginals(max_batch)
                if final_counters is None:
                    final_counters = sam * max_batch
                else:
                    final_counters += sam * max_batch
                iter_ += max_batch
                if verbose:
                    print_batch_step(iter_, max_batch, num_samples)
            iter_ -= max_batch
            if iter_ < num_samples:
                last_batch = num_samples - iter_
                sam = self._extract_marginals(last_batch)
                final_counters += sam * last_batch
                num_done += last_batch
                iter_ += last_batch
                if verbose:
                    print_batch_step(iter_, last_batch, num_samples)
            assert iter_ == num_samples
        if verbose:
            print("\nFinished")

        return final_counters / float(num_done)


def get_linear_layers_path(neighs, out_n, in_n_each, 
                    between_sizes=None,
                    multiply_first_by_in=False):
    """
    """
    if between_sizes is None:
        between_sizes = list()
    try:
        len(between_sizes)
    except:
        raise ValueError("between sizes has to be a list or a tuple")

    layers = {}
    if isinstance(neighs, dict):
        iterat = neighs.values
    else:
        iterat = enumerate(neighs)
    for i, nes in iterat:
        try:
            first = between_sizes[0]*in_n_each if multiply_first_by_in else between_sizes[0]
            in_betw = [first] + list(between_sizes[1:])
        except IndexError:
            in_betw = between_sizes
        layers[i]=[in_n_each*len(nes), *in_betw, out_n]

    return [layers[k] for k in sorted(layers.keys())]