"""
Module: training_utils
Author: Fabio Mazza

Additional stuff for the training
"""
import numpy as np
import torch


def format_src_print(sourc):
    """
    Pretty print all the sources for display
    """
    out = "{ "
    if isinstance(sourc, torch.Tensor):
        x = sourc.cpu().numpy()
    elif not isinstance(sourc,np.ndarray):
        x = np.array(sourc)
    else:
        x = sourc
    for i in range(x.shape[0]):
        out += "{1:3.0f}:{0:01.3f}".format(*x[i])
        if i < x.shape[0]-1:
            out += ", "
    out += "}"
    return out


def print_batch_step(position, batch_size, total):
    """
    Print progress in sampling
    """
    print("Progress {:4.0%}, {:4d} samples, {:5d}".format(
        position/total, batch_size, position), end="\r")

def sample_avg_states(sample_size, net, T):
    sample_raw, x_hat = net.sample(sample_size)
    samples = net.transform_samples(sample_raw)
    S, I, R = calc_avg_states(samples, T)
    return S, I, R


def calc_avg_states(samples, T = 0):
    """
    Calculate average state occurrence in samples
    at time T
    """
    S = samples[:, T, :, 0].mean(0)
    I = samples[:, T, :, 1].mean(0)
    R = samples[:, T, :, 2].mean(0)
    return S, I, R

def make_beta_sequence_three(nbeta, mid=(0.3,0.5), highmid=(0.55,0.8)):
    """
    Construct beta array from 3 piecewise linear
    """
    midn = int(nbeta*mid[0])
    highmidn = int(nbeta*highmid[0])
    
    return np.concatenate((np.linspace(0,mid[1],midn+1)[:-1],
                          np.linspace(mid[1], highmid[1],(highmidn-midn)+1)[:-1],
                          np.linspace(highmid[1],1, nbeta-highmidn+1)[:-1]))


def make_beta_train_exp(nbeta, curv:int = 3, max_x:int=1):
    """
    Make exponential-like curve that ends linearly in 1
    """
    fin_exp = max_x-1/curv

    nexp = int(np.ceil(fin_exp*nbeta/max_x))

    b1 = (1-np.exp(-np.linspace(0,fin_exp, nexp)*curv))

    b2 = np.linspace(b1[-1],1, nbeta-nexp+1)[1:]

    return np.concatenate((b1,b2))
class TrainingPars:
    """
    Class for holding parameters used during the training
    """

    def __init__(self,num_samples,max_step,
                 last_step,softness_log,betas,
                 lr,model, num_sources,
                 print_all_betas=False, rounding_prec=4):
        self.num_samples = num_samples
        self.max_step = max_step
        self.last_step = last_step
        self.softness_log = softness_log
        self.betas = betas
        self.learn_rate = lr
        self.model = model
        self.out_beta_steps = print_all_betas
        self.rounding_prec = rounding_prec
        self.num_sources = num_sources
        self.extra_pars = None

    def add_extra_pars(self, **kwargs):
        """
        Add more parameters by keyword argument
        """
        self.extra_pars = dict(**kwargs)

    def get_for_saving(self):
        """
        Output parameters for saving in JSON
        """
        beta_fin = self.last_step > 0
        betas = self.betas
        if len(betas) < 100 and (not self.out_beta_steps):
            betas_out = [round(v, self.rounding_prec) for v in betas]
            if beta_fin:
                betas_out.append("1.0-final")
        else:
            betas_out = {
                "start": betas[0],
                "stop": betas[-1],
                "step": betas[1] - betas[0],
                "num_beta": len(betas)
            }
            if beta_fin:
                betas_out["beta_final"] = 1

        out_dict = {
            "num_samples": self.num_samples,
            "max_step": self.max_step,
            "last_step": self.last_step,
            "softness_log": self.softness_log,
            "learning_rate": self.learn_rate,
            "num_sources": self.num_sources,
            "betas": betas_out
        }
        if self.extra_pars is not None:
            out_dict.update(self.extra_pars)

        return out_dict


def sort_I(M, t=0, ):
    """
    Get marginal probability at time t of being infected
    and sort the index by it
    """
    if not torch.is_tensor(M):
        M_I = torch.Tensor(M[:, t, 1])
    else:
        M_I = M[:, t, 1]
    prob, index = M_I.sort(descending=True)
    return torch.stack((prob, index.type(prob.type())), dim=1).squeeze()