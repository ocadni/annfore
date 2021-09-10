from enum import Enum

import numpy as np
import pandas as pd
import torch


def batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).

    Used for the calculation of the energy
    Thanks to https://github.com/pytorch/pytorch/issues/14489
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)

class Capabil(Enum):
    """
    Energy model capabilities
    """
    ENERGY_SEP = 1

class EnergyModel:
    """
    Abstract Energy Model class
    """

    def __init__(self, device, dtype, capabilities):
        self.capabil = frozenset(capabilities)
        self.device = torch.device(device)
        self.dtype = dtype

        self.count_zero = -1
        

    def get_capabilities(self):
        """
        Get model capabilities
        """
        return self.capabil

    def params(self):
        """
        Return used parameters
        """
        raise NotImplementedError

    def energy_(self, samples, debug=False):
        """
        Compute energy of the samples
        """
        raise NotImplementedError

    def get_empty_matrix(self, dims, data_type="undef"):
        if data_type == "undef":
            data_type = self.dtype
        return torch.zeros(dims, dtype=data_type, device=self.device)


def calc_p_no_c_np(contacts,states,T):
    """
    Function to check the calculation of the p_no_contacts
    """
    contacts_pd = pd.DataFrame(contacts,columns=["t","i","j","lam"])
    p_out = []
    for t in range(T-1):
        status = states[t]
        c_t = contacts_pd[contacts_pd.t==t].copy()
        c_t["pnc"]=(1 - c_t.lam * status[c_t.j.astype(int)])

        p_no_c = np.ones(status.shape,dtype=np.float)

        for index,l in c_t.iterrows():
            p_no_c[int(l.i)] *= l.pnc
        p_out.append(p_no_c)
    return np.stack(p_out)



def check_convert_tensor(value, N, dtype, device):
    try:
        f = len(value)
        assert f == N
        values = torch.Tensor(value, dtype=dtype, device=device)
        vals_zero = torch.all(values == 0).item()
    except TypeError:
        # the mu is not an array
        if isinstance(value, float) or isinstance(value, int):
            vals_zero = (value == 0)
            values = torch.full((N,), value, dtype=dtype, device=device)
        else:
            raise ValueError()
    return values, vals_zero

def calc_psus_masks(masks, N, t_limit):
    """
    Calculate the psus from the masks
    """
    
    assert masks.shape == (N,2,t_limit+2)

    ntimes = np.prod(masks.sum(-1) -1,-1)
    ### check the nodes that can be infected
    correct_times = ntimes[masks[...,0,-1]]

    return 1 - 1./(2 + correct_times.sum()/N)

def calc_psus_masks_newavg(masks, N, t_limit):
    ntimes = np.prod(masks.sum(-1) -1,-1)
    correct_times = ntimes[masks[...,0,-1]]
    print(correct_times)
    g = (1 - 1./(2 + correct_times))
    return g.sum()/N

def calc_psus_masks_nodes(masks,N, t_limit, limzero=1e-30): 
    assert masks.shape == (N,2,t_limit+2)

    ntimes = np.prod(masks.sum(-1),-1)
    ### check the nodes that can be infected
    are_prossiblyS = masks[...,0,-1]
    
    psus = 1 - 1. / ntimes
    # these are certain to be S
    psus[ntimes==1] = 0.5
    # put the ones who cannot be S
    psus[np.logical_not(are_prossiblyS)] = 0.5

    return psus

def calc_pendst_masks(masks, N, t_limit):
    """
    Calculate the probabilities of the final state
    Give the masks and N, t_limit for sanity check
    """
    assert masks.shape == (N,2,t_limit+2)
    ntimes=masks.sum(-1)

    last_t = masks[...,-1].astype(int)
    p_s = np.prod(last_t,-1)/np.prod(ntimes,-1)
    p_i=(ntimes[...,0]-last_t[...,0])*last_t[...,1]/(np.prod(ntimes,-1))
    p_r=np.prod((ntimes-last_t),-1)/np.prod(ntimes,-1)
    probs_end = np.stack((p_s,p_i,p_r))

    return probs_end

def calc_extra_entr(pfinstates, N):
    assert pfinstates.shape == (3,N)
    mlogp = -np.ma.log(pfinstates)

    mul = mlogp.max(axis=0).data - mlogp

    return mul.data