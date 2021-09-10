import time
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from . import common as commod
from annfore.utils.graph import find_neighs
from annfore.utils.conf_utils import one_hot_samples, one_hot_samples_relative_r

check_convert_tensor = commod.check_convert_tensor
calc_psus_masks = commod.calc_psus_masks

calc_psus_masks_newavg = commod.calc_psus_masks_newavg
calc_psus_masks_nodes = commod.calc_psus_masks_nodes

calc_pendst_masks = commod.calc_pendst_masks

calc_extra_entr = commod.calc_extra_entr


class SirModel(commod.EnergyModel):
    """
    Energy model with probabilities
    If you put p_rec_t_0 (!=None),
    it will consider p_source as the probability
    of x_i^{t=0} in state I
    """
    def __init__(self, 
                 contacts, 
                 mu = 0,
                 delta = 0,
                 device = "cpu",
                 dtype = torch.float,
                 p_source = 1e-6,
                 p_rec_t_0 = None,
                 p_sus=(0.5,0),
                 p_obs = 1e-10,
                 p_w = 1e-10,
                 q = 3,
                 continuous = False,
                 err_max_lambda = 1e-6):
        
        super().__init__(device, dtype, [])
        self.N = int(max(contacts[:, 1]) + 1)
        self.T = int(max(contacts[:, 0]) + 2)
        self.nt = int(self.N * self.T)
        self.count_zero = 0
        # Number of states for a node
        self.q = q
        self.err_max_lambda = err_max_lambda
        self.dtype = dtype
        self.device = torch.device(device)
        
        self.p_source = p_source
        if p_rec_t_0 is None:
            self.p_rec_t_0 = p_source
            self.p_sus_t_0 = 1 - p_source
        else:
            self.p_rec_t_0 = p_rec_t_0
            self.p_sus_t_0 = 1 - p_source - p_rec_t_0
        
        #assert(self.p_source + self.p_rec_t_0 <= 1)
        self.p_sus = p_sus
        self.p_obs = p_obs
        self.p_w = p_w
        
        self.delta = delta
        self.mu = mu
        self.mu_cont = 0
        if continuous:
            self.mu_cont = mu
        
        self.log_obs = self.get_empty_matrix((self.N, self.T, self.q))
        
        self.neighs = find_neighs(contacts, N = self.N)
        self.create_lambdas_tensor(contacts)
        self.extra_params = {}

    def params(self):
        out_dic ={
            "N": self.N,
            "T" : self.T,
            "p_source": self.p_source,
            "p_rec_t_0": self.p_rec_t_0,
            "p_obs" : self.p_obs,
            "p_w": self.p_w,
            "mu":self.mu,
            "p_sus_time": None if self.p_sus is None else self.p_sus[1],
            "p_sus_value": None if self.p_sus is None else self.p_sus[0]
        }
        out_dic.update(self.extra_params)
        return out_dic

    def get_empty_matrix(self, dims, data_type="undef"):
        if data_type == "undef":
            data_type = self.dtype
        return torch.zeros(dims, dtype=data_type, device=self.device)

    def create_lambdas_tensor(self, contacts):
        """
        Creates the matrix for the contacts, shape (T,N,N)
        """
        T = self.T
        N = self.N
        neighs = self.neighs
        self.logp_lam = {}
        for n in range(N):
            self.logp_lam[n] = self.get_empty_matrix((len(neighs[n]), T))
        for cc in contacts:
            t = int(cc[0])
            if t >= T:
                raise ValueError("Contact time above T!")
            i = int(cc[1])
            j = int(cc[2])
            lam = cc[3]
            lam = np.clip(lam, 0, 1 - self.err_max_lambda)
            #print(t,i,j,lam)
            index_i = neighs[j].index(i)
            if self.logp_lam[j][index_i][t] > 0:
                print(f"multiple contacts between {i} -> {j} at time {t}")
            self.logp_lam[j][index_i][t] = np.log1p(-lam)
    
    def create_deltas_tensor(self, deltas):
        """
        Delta time of interactions between indviduals (used when working with rate of transmissions)
        """
        T = self.T
        N = self.N
        neighs = self.neighs
        self.deltas = {}
        for n in range(N):
            self.deltas[n] = self.get_empty_matrix((len(neighs[n]), T))
        for cc in deltas:
            t = int(cc[0])
            if t >= T:
                raise ValueError("Contact time above T!")
            i = int(cc[1])
            j = int(cc[2])
            delta = cc[3]
            #lam = np.clip(lam, 0, 1 - self.err_max_lambda)
            #print(t,i,j,lam)
            index_i = neighs[j].index(i)
            self.deltas[j][index_i][t] = delta

        '''def create_delta_tensor(self, gamma):
            """
            Deltas values for the computation of parameters of rate of contagion
            """
            N = self.N
            self.deltas = {}
            for n in range(N):
                self.deltas[n] = self.logp_lam[n]/gamma
    '''
        
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

    def set_obs(self, obs_list):
        '''
        list of observations:
        (node_obs, t_obs, state_obs)
        
        state_obs = (0,1,2, ... , q-1)
        '''
        
        for n_obs, t_obs, q_obs in obs_list:
            if n_obs >= self.N:
                raise ValueError(f"node not present, obs:({n_obs},{t_obs},{q_obs}), {n_obs} >= {self.N}")
            if t_obs >= self.T:
                raise ValueError(f"time obs larger than T,  obs:({n_obs},{t_obs},{q_obs}), {t_obs} >= {self.T}")
            for q in range(self.q):
                self.log_obs[n_obs, t_obs, q] = 1 - int(q == q_obs)
            
    def set_p_source(self, p_source):
        """
        Set the P_source for the single source constraint
        """
        self.p_source = p_source
        warnings.warn("Setting both source and recovery at t=0 probabilities")
        self.p_rec_t_0 = p_source
        self.p_sus_t_0 = 1

        return True
    
    def set_sources_probs(self, p_source, p_rec_t_0):
        """
        Set both p_source and p_rec_t_0
        """
        self.p_source = p_source
        self.p_rec_t_0 = p_rec_t_0
        self.p_sus_t_0 = 1- p_source

    def P_no_trasm_i(self, node_i, x):
        neighs_i = self.neighs[node_i]
        m = x.shape[0]        
        if(len(neighs_i) > 0):
            x_select = x[:,neighs_i,:]
            x_select_hot = one_hot_samples(x_select, self.T, q=self.q)
            res_m = (x_select_hot[:,:,:,1] * self.logp_lam[node_i]).sum(dim=1)
        else:
            # the input vector should have dimension zero
            res_m = torch.zeros((m, self.T), device=self.device)

        return torch.exp(res_m)

    
    def energy_i(self, node_i, x):
        assert x.shape[1:] == (self.N, self.q-1)
                
        P_no_trasm_i = self.P_no_trasm_i(node_i, x)
        
        x_i_hot = one_hot_samples(x[:,[node_i], :], self.T, q = self.q).squeeze()
        
        S = x_i_hot[:,:,0]
        I = x_i_hot[:,:,1]
        R = x_i_hot[:,:,2]
        
        w = torch.zeros_like(x_i_hot, device=self.device, dtype=self.dtype)
        w[:,:,0] = P_no_trasm_i * S + self.delta * I
        w[:,:,1] = (1. - P_no_trasm_i) * S * (1. - self.mu_cont) + (1. - self.delta - self.mu) * I
        w[:,:,2] = (1. - w[:,:,0] - w[:,:,1]).clamp(0)
                
        # Computing log of P_MC
        P_w = w[:,:-1] * x_i_hot[:,1:]
        P_w = P_w.sum(dim=2)
        self.count_zero += int((P_w==0).sum().item())
        #P_w[P_w == 0] = self.p_w
        #P_w = torch.max(self.p_w, P_w)
        P_w.clamp_(self.p_w)
        log_P_w = torch.log(P_w)
        #print(log_P_w)
        log_P_w_tot = log_P_w.sum(dim=1)
        
        # computing log of constraint obs
        log_P_obs = (x_i_hot * self.log_obs[node_i,:]).sum((1,2))*np.log(self.p_obs)
        
        #computing log of source bias
        log_P_sources = torch.log(self.p_sus_t_0*S[:,0] + self.p_source*I[:,0]+self.p_rec_t_0*R[:,0])

        #computing log weighted fraction S/I
        if self.p_sus is not None:
            log_p_sus = self._log_psus_i(node_i, S, I, R)
        else:
            log_p_sus = torch.zeros_like(log_P_sources)
        
        #print(log_P_w_tot, log_P_obs, log_P_sources)
        
        ener = (-log_P_w_tot, -log_P_obs, -log_P_sources, -log_p_sus)

        return ener

    def _log_psus_i(self, node_i, S, I, R):
        p_sus = self.p_sus[0]
        p_inf = 1 - p_sus
        p_rec = 1 - p_sus
        t_p_sus=int(self.p_sus[1])
        #print(f"{S[:,t_p_sus]} {self.p_sus[0]}")
        #print(f"{I[:,t_p_sus]} {p_inf}")
        #print(f"{R[:,t_p_sus]} {p_rec}")
        return torch.log(p_sus * S[:,t_p_sus]+
                            p_inf * I[:,t_p_sus]+
                            p_rec * R[:,t_p_sus])

    def energy_separated(self, x):
        self.count_zero = 0

        E_w_tot, E_obs, E_sources, E_sus = self.energy_i(0, x)
        for node_i in range(1, self.N):
            #ener_list=self.energy_i(node_i, x)
            E_w_tot_i, E_obs_i, E_sources_i, E_sus_i = self.energy_i(node_i, x)
            E_w_tot+=E_w_tot_i
            E_obs+=E_obs_i
            E_sources+=E_sources_i
            E_sus+=E_sus_i

        return E_w_tot, E_obs, E_sources, E_sus

    def energy(self, x):
        return sum(self.energy_separated(x))
    
    def energy_(self, x, softness_log=0):
        return self.energy(x)

class SirModel_susSep(SirModel):
    def __init__(self, 
                 contacts, 
                 mu = 0,
                 delta = 0,
                 device = "cpu",
                 dtype = torch.float,
                 p_source = 1e-6,
                 p_rec_t_0 = None,
                 p_sus=(0.5,0),
                 p_obs = 1e-10,
                 p_w = 1e-10,
                 q = 3,
                 continuous = False,
                 err_max_lambda = 1e-6):

        super().__init__(contacts, mu=mu, delta=delta, device=device,
                dtype=dtype, p_source=p_source, p_rec_t_0=p_rec_t_0, p_sus=p_sus, p_obs=p_obs,
                p_w=p_w, q=q, continuous=continuous, err_max_lambda=err_max_lambda)
        
        self.p_sus_mat = p_sus[0]
        self.p_sus = (p_sus[0].mean(), p_sus[1])

    def _log_psus_i(self, node_i, S, I, R):
        p_sus = self.p_sus_mat[node_i]
        p_inf = 1 - p_sus
        p_rec = 1 - p_sus
        t_p_sus=int(self.p_sus[1])
        #print(f"{S[:,t_p_sus]} {self.p_sus[0]}")
        #print(f"{I[:,t_p_sus]} {p_inf}")
        #print(f"{R[:,t_p_sus]} {p_rec}")
        return torch.log(p_sus * S[:,t_p_sus]+
                            p_inf * I[:,t_p_sus]+
                            p_rec * R[:,t_p_sus])

class SirModel_bal(SirModel):

    def __init__(self, 
                 contacts, 
                 mu = 0,
                 delta = 0,
                 device = "cpu",
                 dtype = torch.float,
                 p_source = 1e-6,
                 p_rec_t_0 = None,
                 p_sus=(0.5,0),
                 p_obs = 1e-10,
                 p_w = 1e-10,
                 q = 3,
                 continuous = False,
                 err_max_lambda = 1e-6):

        super().__init__(contacts, mu=mu, delta=delta, device=device,
                dtype=dtype, p_source=p_source, p_rec_t_0=p_rec_t_0, p_sus=p_sus, p_obs=p_obs,
                p_w=p_w, q=q, continuous=continuous, err_max_lambda=err_max_lambda)

        self.masks=None
        self.probs_fstate=None
        self.use_psus_too = False
        self.lpsus_energy = None
    
    def set_forced_psus(self, psus_val=None, multipl=1, use_psus_energy=False):
        self.use_psus_too = True
        #psus = self.p_sus[0]
        if self.masks is None:
            raise ValueError("Call `set_masks` before this method")
        canbeS = (self.probs_fstate[0] > 0) & (self.probs_fstate[0] < 1-1e-12)

        if psus_val is None:
            psus = calc_psus_masks_nodes(self.masks, self.N, self.T-1)
            hI_R = (np.log(1-psus)-np.log(psus))*multipl
            val_logp = hI_R[canbeS]
        else:
            psus = psus_val
            val_logp = (np.log(1-psus_val)-np.log(psus_val))
        extra_logp = ((self.probs_fstate[1:,canbeS] > 0) * val_logp).T
        if use_psus_energy:
            self.lpsus_energy = extra_logp
            print(extra_logp.shape)
        else:
            ## adjust probabilities
            self.extra_pflog[canbeS, 1:] += extra_logp
    
    def set_masks(self, masks, logmult=1):
        self.masks = masks
        self.probs_fstate = calc_pendst_masks(masks, self.N, self.T-1)
        ## this is LOG(P), not -LOG(P)
        self.extra_pflog = -calc_extra_entr(self.probs_fstate, self.N).T * logmult
        #self.psus = "Det"

    def _log_psus_i(self, node_i, S, I, R):
        try:
            vals = self.extra_pflog[node_i]
        except AttributeError:
            return torch.zeros(S.shape[0], dtype=S.dtype, device=S.device)
        return S[:,-1] * vals[0] + I[:,-1]*vals[1] + R[:,-1]*vals[2]

    def energy_i(self, node_i, x):
        assert x.shape[1:] == (self.N, self.q-1)
                
        P_no_trasm_i = self.P_no_trasm_i(node_i, x)
        
        x_i_hot = one_hot_samples(x[:,[node_i], :], self.T, q = self.q).squeeze()
        
        S = x_i_hot[:,:,0]
        I = x_i_hot[:,:,1]
        R = x_i_hot[:,:,2]
        
        w = torch.zeros_like(x_i_hot, device=self.device, dtype=self.dtype)
        w[:,:,0] = P_no_trasm_i * S + self.delta * I
        w[:,:,1] = (1. - P_no_trasm_i) * S * (1. - self.mu_cont) + (1. - self.delta - self.mu) * I
        w[:,:,2] = (1. - w[:,:,0] - w[:,:,1]).clamp(0)
                
        # Computing log of P_MC
        P_w = w[:,:-1] * x_i_hot[:,1:]
        P_w = P_w.sum(dim=2)
        self.count_zero += int((P_w==0).sum().item())
        #P_w[P_w == 0] = self.p_w
        #P_w = torch.max(self.p_w, P_w)
        P_w.clamp_(self.p_w)
        log_P_w = torch.log(P_w)
        #print(log_P_w)
        log_P_w_tot = log_P_w.sum(dim=1)
        
        # computing log of constraint obs
        log_P_obs = (x_i_hot * self.log_obs[node_i,:]).sum((1,2))*np.log(self.p_obs)
        
        #computing log of source bias
        x0 = x_i_hot[:,0,:]
        log_P_sources = torch.log(self.p_sus_t_0*S[:,0] + self.p_source*I[:,0]+self.p_rec_t_0*R[:,0])

        #computing log weighted fraction S/I
        if self.p_sus is not None:
            log_p_extra = self._log_psus_i(node_i, S, I, R)
        else:
            log_p_extra = torch.zeros_like(log_P_sources)

        if self.lpsus_energy is not None:
            vals = self.extra_pflog[node_i]
            log_P_sources += (I[:,-1]*vals[0] + R[:,-1]*vals[1])
        
        #print(log_P_w_tot, log_P_obs, log_P_sources)
        
        ener = (-log_P_w_tot, -log_P_obs, -log_P_sources, -log_p_extra)

        return ener


class SirModel_norm(SirModel):
    def set_psus(self, psus=0.5):
        self.p_sus = psus
        
    def P_no_trasm_i(self, node_i, x):
        neighs_i = self.neighs[node_i]
        m = x.shape[0]        
        lam_i = 1 - torch.exp(self.logp_lam[node_i])
        if(len(neighs_i) > 0):
            x_select = x[:,neighs_i,:]
            x_select_hot = one_hot_samples(x_select, self.T, q=self.q)
            res_m = (1 - x_select_hot[:,:,:,1] * lam_i).prod(dim=1)
        else:
            # the input vector should have dimension zero
            res_m = torch.ones((m, self.T), device=self.device)

        return res_m

    def energy_i(self, node_i, x):
        assert x.shape[1:] == (self.N, self.q-1)
                
        P_no_trasm_i = self.P_no_trasm_i(node_i, x)
        
        x_i_hot = one_hot_samples(x[:,[node_i], :], self.T, q = self.q).squeeze()
        
        S = x_i_hot[:,:,0]
        I = x_i_hot[:,:,1]
        #R = x_i_hot[:,:,2] ## TODO: check why we don't need the R state
        norm = 1.+ 3.*self.p_w
        w = torch.zeros_like(x_i_hot, device=self.device, dtype=self.dtype)
        w[:,:,0] = P_no_trasm_i * S + self.delta * I
        w[:,:,1] = (1. - P_no_trasm_i) * S + (1. - self.delta - self.mu) * I
        w[:,:,2] = torch.max(1. - w[:,:,0] - w[:,:,1], torch.zeros_like(w[:,:,0]))
        w[:,:,:] += self.p_w
        w[:,:,:] /= norm
        
        # Computing log of P_MC
        P_w = w[:,:-1] * x_i_hot[:,1:]
        P_w = P_w.sum(dim=2)
        self.count_zero += len(P_w[P_w==0])
        #P_w[P_w == 0] = self.p_w
        log_P_w = torch.log(P_w)
        #print(log_P_w)
        log_P_w_tot = log_P_w.sum(dim=1)
        
        # computing log of constraint obs
        log_P_obs = (x_i_hot * self.log_obs[node_i,:]).sum((1,2))*np.log(self.p_obs)
        
        #computing log of source bias
        x0 = x_i_hot[:,0,:]
        log_P_sources = torch.log(self.p_sus_t_0 * x0[:,0] + self.p_source*x0[:,1]+self.p_rec_t_0*x0[:,2])
        
        #log_P_sus = torch.log(self.p_sus*x_i_hot[:,:,0] + x_i_hot[:,:,1] + x_i_hot[:,:,2]).sum(dim=1)

        ener = -(log_P_w_tot + log_P_obs + log_P_sources)
        #print(log_P_w_tot, log_P_obs, log_P_sources)
        

        return ener
 
    

class SirModel_relative_r(SirModel):
    def P_no_trasm_i(self, node_i, x):
        neighs_i = self.neighs[node_i]
        m = x.shape[0]        
        if(len(neighs_i) > 0):
            x_select = x[:,neighs_i,:]
            x_select_hot = one_hot_samples_relative_r(x_select, self.T, q=self.q)
            res_m = (x_select_hot[:,:,:,1] * self.logp_lam[node_i]).sum(dim=1)
        else:
            # the input vector should have dimension zero
            res_m = torch.zeros((m, 1), device=self.device)

        return torch.exp(res_m)
    def energy_i(self, node_i, x):
        assert x.shape[1:] == (self.N, self.q-1)
                
        P_no_trasm_i = self.P_no_trasm_i(node_i, x)
        
        x_i_hot = one_hot_samples_relative_r(x[:,[node_i], :], self.T, q = self.q).squeeze()
        
        S = x_i_hot[:,:,0]
        I = x_i_hot[:,:,1]
        R = x_i_hot[:,:,2]
        
        w = torch.zeros_like(x_i_hot, device=self.device, dtype=self.dtype)
        w[:,:,0] = P_no_trasm_i * S + self.delta * I
        w[:,:,1] = (1. - P_no_trasm_i) * S + (1. - self.delta - self.mu) * I
        w[:,:,2] = self.mu * I + R
                
        # Computing log of P_MC
        P_w = w[:,:-1] * x_i_hot[:,1:]
        P_w = P_w.sum(dim=2)
        P_w[P_w == 0] = self.p_w
        log_P_w = torch.log(P_w)
        #print(log_P_w)
        log_P_w_tot = log_P_w.sum(dim=1)
        
        # computing log of constraint obs
        log_P_obs = (x_i_hot * self.log_obs[node_i,:]).sum((1,2))*np.log(self.p_obs)
        
        #computing log of source bias
        x0 = x_i_hot[:,0,:]
        log_P_sources = torch.log(self.p_sus_t_0*x0[:,0] + self.p_source*x0[:,1]+self.p_rec_t_0*x0[:,2])
        
        #print(log_P_w_tot, log_P_obs, log_P_sources)
        
        ener = -(log_P_w_tot + log_P_obs + log_P_sources)

        return ener
