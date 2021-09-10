
"""
SIR model
Needed for the energy calculation of the epidemic cascades

Authors: Indaco Biazzo, Fabio Mazza
"""
import numpy as np
import torch
import torch.nn.functional as F

from models.common import EnergyModel
from models.common import Capabil



def check_convert_tensor(value, N, dtype, device):
    """
    Convenience function to check python types
    """
    try:
        f = len(value)
        assert f == N
        values = torch.Tensor(value, dtype=dtype, device=device)
        vals_zero = torch.all(values == 0).item()
    except TypeError as t_error:
        # the mu is not an array
        if isinstance(value, float) or isinstance(value, int):
            vals_zero = (value == 0)
            values = torch.full((N,), value, dtype=dtype, device=device)
        else:
            raise ValueError from t_error
    return values, vals_zero

def transform_to_spins(bin_tensor):
    if torch.any(bin_tensor > 1) or torch.any(bin_tensor < 0):
        return ValueError("Tensor must be made of zeros and ones")
    return bin_tensor*2 - 1

def normalize(tensor, dim=None):
    return tensor.float() / tensor.sum()


class SirModel(EnergyModel):
    def __init__(self, 
                 contacts, 
                 mu = 0,
                 delta = 0,
                 device = "cpu",
                 dtype = torch.float,
                 h_source = 3,
                 h_obs = 10,
                 h_log_p = 10,
                 q = 3,
                 num_sources=1,
                 log_softness = 1e-8,
                 sparse_lambdas=None):

        super().__init__(device, dtype, (Capabil.ENERGY_SEP,))

        self.N = int(max(contacts[:, 1]) + 1)
        self.T = int(max(contacts[:, 0]) + 2)
        self.nt = int(self.N * self.T)
        # Number of states for a node
        self.q = q
        self.delta = delta
        self.mu = mu

        self.h_source = h_source
        self.h_obs_value = h_obs
        self.h_log_p = h_log_p
        self.err_log_p = np.exp(-h_log_p)

        self.log_softness = log_softness
        self.num_sources = num_sources
        print("creating log_obs and bias")
        self.log_obs = self.get_empty_matrix((self.T, self.N, self.q))
        self.log_sources_bias = self.get_empty_matrix((self.N, self.q))
        print("creating lambdas tensor")
        if sparse_lambdas is not None:
            self.sparse_lambdas = sparse_lambdas
        elif self.N < 200:
            self.sparse_lambdas = False
        else:
            self.sparse_lambdas = True
        if self.sparse_lambdas == True:
            print("Using sparse tensor for lambdas")

        self.create_lambdas_tensor(contacts,self.sparse_lambdas)
        #else:
        #    self.create_lambdas_tensor_sparse(contacts)
        print("set bias sources")
        self.set_sources(num_sources, h_source)
        self.parameters = {
            "N": self.N,
            "T" : self.T,
            "num_sources": self.num_sources,
            "h_obs" : self.h_obs_value,
            "h_source": self.h_source,
            "h_mc" : self.err_log_p,
        }

    def set_sources(self, n_sources, h_src):
        """
        Set the sources matrix:
        since we don't want R states, we need h_src
        """
        self.num_sources = n_sources
        self.h_source = h_src

        log_p_inf = np.log(n_sources / self.N)
        log_p_sus = np.log(1 - (n_sources)/self.N)

        self.log_sources_bias[:,0] = log_p_sus
        self.log_sources_bias[:,1] = log_p_inf
        self.log_sources_bias[:,2] = - h_src


    def create_lambdas_tensor(self, contacts,sparse=False):
        """
        Creates the matrix for the contacts, shape (T,N,N)
        """
        T = self.T
        N = self.N
        
        logp_lambdas = np.log(
            np.maximum(1-contacts[:,-1],self.log_softness)
        )
        
        indics = torch.LongTensor(contacts[:,:-1]).t().to(self.device)
        if sparse:
            values = torch.FloatTensor(logp_lambdas).to(self.device)
            self.logp_lam = torch.sparse.FloatTensor(indics,values,
                                                     torch.Size([T,N,N],
                                                     device=self.device,
                                                     dtype=self.dtype))
        else:
            tens = self.get_empty_matrix((T,N,N))
            tens[indics[0],indics[1],indics[2]] = torch.tensor(logp_lambdas,dtype=self.dtype).to(self.device)
            self.logp_lam  = tens


    def create_set_deltas(self, delta):
        try:
            self.deltas, self.deltas_zero = check_convert_tensor(
                delta, self.N, self.dtype, self.device)
        except ValueError:
            raise ValueError("Insert number or array for delta")

    def create_set_mu(self, mu):
        try:
            self.mus, self.mus_zero = check_convert_tensor(
                mu, self.N, self.dtype, self.device)
        except ValueError:
            raise ValueError("Insert number or array of mus")
    

    def set_last_obs(self, last_obs, h_obs="not_set"):
        """
        Set the values of the observation matrix for the last time instance
        """
        
        # transform obs
        if len(last_obs) != self.N:
            raise ValueError("last_obs has the wrong size")
        if not isinstance(last_obs, torch.Tensor):
            last_obs = torch.tensor(last_obs, dtype=torch.long)

        if h_obs != "not_set":
            self.h_obs_value = h_obs

        obs_one_hot = F.one_hot(last_obs.long(),self.q)
        for n in range(obs_one_hot.shape[0]):
            if obs_one_hot[n].sum() > 0:
                self.log_obs[-1, n, :] = (obs_one_hot[n]-1)
        
        return True

    def set_log_obs(self, obs, h_obs="not_set"):
        """
        Problems setting obs: which are the valid obs?
        If 0 => S, 1 => I, 2 => R?
        """
        
        # transform obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.long)
        if obs.shape != torch.Size((self.T,self.N, self.q)):
            raise ValueError("obs has the wrong size, should be TxN Tensor")

        if h_obs != "not_set":
            self.h_obs_value = h_obs

        self.log_obs = obs
        return True


    def set_h_log_p(self, h_log_p):
        self.h_log_p = h_log_p
        self.err_log_p = np.exp(-h_log_p)
        return True

    def params(self):
        return {
            "N": self.N,
            "T" : self.T,
            "num_sources": self.num_sources,
            "h_obs" : self.h_obs_value,
            "h_source": self.h_source,
            "h_mc" : self.err_log_p,
        }
    

    def calc_log_P_no_c(self,samples_hot):
        """
        Calculate log of the probability of no contact
        """
        if self.sparse_lambdas:
            return self.calc_log_P_no_c_sparse(samples_hot)
        else:
            return torch.matmul(self.logp_lam.unsqueeze(0), 
                            samples_hot[:,:,:,1].unsqueeze(3)).squeeze()

    def calc_log_P_no_c_sparse(self,samples_hot):
        """
        Calculate log of the probability of no contact, with sparse 
        log lambda matrix
        """
        r = torch.stack([torch.mm(self.logp_lam[t].to_dense(),samples_hot[:,t,:,1].t()) for t in range(self.T)] )
        return r.permute(2,0,1)

    def calc_log_mc_prob(self, samples_hot):
        """
        Calculate log prob of markov chain
        """

        # samples have to be in shape [num_samples,T,N] with values 0,1,2
        assert samples_hot.shape[1:] == (self.T,self.N,self.q)
        if samples_hot.max() >= 2 or samples_hot.min() < 0:
            raise ValueError("Samples have to be between 0 and 1")

        m = samples_hot.shape[0]
        # no need to convert to one_hot => dimensions  are already [m,T,N,q]
        #samples = F.one_hot(samples.long(), self.q).float()
        
        ## contribution from the transition probabilities, only for t > 0
        # Calculate matrix of transition A
        P_no_c = self.calc_log_P_no_c(samples_hot).exp()

        # Calculate probability due to the Markov Chain contribution 
        P_MC = torch.zeros_like(samples_hot)
        S = samples_hot[:,:,:,0]
        I = samples_hot[:,:,:,1]
        R = samples_hot[:,:,:,2]
        P_MC[:,:,:,0] = P_no_c * S + self.delta * I
        P_MC[:,:,:,1] = (1. - P_no_c) * S + (1. - self.delta - self.mu) * I
        P_MC[:,:,:,2] = self.mu * I + R
        
        # shifting of one time foward and setting to 1 the T = 0
        P_MC = P_MC[:,:-1,:,:]
        
        # Computing log of P_MC
        P_MC_samples = P_MC * samples_hot[:,1:]
        P_MC_samples = P_MC_samples.sum(dim=3).clamp_min(self.err_log_p)
        log_P_MC = torch.log(P_MC_samples)
        log_P_MC_tot = log_P_MC.sum((1,2))
        return log_P_MC, log_P_MC_tot
    
    def energy_(self, samples_hot, debug=False):
        """
        Compute energy of the samples, have to be one hot encoded
        """
        #print(log_P_bias_sources.shape)
        mlog_P_MC_tot, mlog_P_obs, mlog_P_bias_sources = self.energy_separated(
            samples_hot, debug)
        ener = mlog_P_MC_tot + mlog_P_obs+ mlog_P_bias_sources

        return ener.to(device=self.device,dtype=self.dtype)


    def energy_separated(self, samples_hot, debug=None):
        """
        Calculate energy of the samples, giving in the order
        energy MC, energy OBS, energy SOURCES
        """

        # samples have to be in shape [num_samples,T,N] with values 0,1,2
        assert samples_hot.shape[1:] == (self.T,self.N,self.q)
        if samples_hot.max() >= 2 or samples_hot.min() < 0:
            raise ValueError("Samples have to be between 0 and 1")

        num_samples = samples_hot.shape[0]
        # no need to convert to one_hot => dimensions  are already [m,T,N,q]
        #samples = F.one_hot(samples.long(), self.q).float()
        
        ## contribution from the transition probabilities, only for t > 0
        # Calculate matrix of transition A
        logP_no_c = self.calc_log_P_no_c(samples_hot)
        P_no_c = torch.exp(logP_no_c)   

        # Calculate probability due to the Markov Chain contribution 
        P_MC = torch.zeros_like(samples_hot)
        S = samples_hot[:,:,:,0]
        I = samples_hot[:,:,:,1]
        R = samples_hot[:,:,:,2]
        P_MC[:,:,:,0] = P_no_c * S + self.delta * I
        P_MC[:,:,:,1] = (1. - P_no_c) * S + (1. - self.delta - self.mu) * I
        P_MC[:,:,:,2] = self.mu * I + R
        
        # shifting of one time foward and setting to 1 the T = 0
        P_MC = P_MC[:,:-1,:,:]
        
        # Computing log of P_MC
        P_MC_samples = P_MC * samples_hot[:,1:]
        P_MC_samples = P_MC_samples.sum(dim=3)
        P_MC_samples[P_MC_samples == 0] = self.err_log_p
        log_P_MC = torch.log(P_MC_samples)
        log_P_MC_tot = log_P_MC.sum((1,2))
        
        # computing log of constraint obs
        log_P_obs = (samples_hot * self.log_obs).sum((1,2,3)) * self.h_obs_value
        #print(log_P_obs.shape)
        
        #computing log of source bias
        log_P_bias_sources = (samples_hot[:,0,:,:] * self.log_sources_bias).sum((1,2))

        return -log_P_MC_tot, -log_P_obs, -log_P_bias_sources

