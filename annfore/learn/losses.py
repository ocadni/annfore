import time
import torch
import numpy as np
from ..models.common import Capabil
import pandas as pd

#from ..learn.l_utils import 

ENERGIES_NAMES = ["ener_mc", "ener_obs", "ener_sources", "ener_p_sus"]

def loss_fn_coeff(net, model, beta):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        with torch.no_grad():

            times = {}
            last_time, start_t = time.time(), time.time()
            #with torch.no_grad():
            samples, probs = net.sample(num_samples)
            last_time, times["sample"] = time.time(), time.time() - last_time
            log_prob = net._log_prob(samples, probs)
            last_time, times["log_prob"] = time.time(), time.time() - last_time
            energies_separ = Capabil.ENERGY_SEP in model.get_capabilities()
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            if energies_separ:
                energies = model.energy_separated(samples)
                loss_pre_info = {enr_name: e.mean().item() for enr_name, e in zip(ENERGIES_NAMES, energies)}
                energy = sum(energies)
            else:
                loss_pre_info = {}
                energy = model.energy_(samples)
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            loss = log_prob + beta * energy
            loss_exact = log_prob + energy
            last_time, times["loss"] = time.time(), time.time() - last_time
            loss -= loss.mean()
        loss_info = {"energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "times": times}

        loss_info.update(loss_pre_info)
        
        return samples, loss, loss_info
    return loss
"""
        c_mult = 1e-6
        c_fin = 0.1/(1+c_mult)
        beta_mult = lambda b: 0.1/(b+c_mult) -c_fin
"""
def loss_fn_coeff_p_sus(net, model, beta, fun_beta_inv=None):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        with torch.no_grad():

            times = {}
            last_time, start_t = time.time(), time.time()
            #with torch.no_grad():
            samples, probs = net.sample(num_samples)
            last_time, times["sample"] = time.time(), time.time() - last_time
            log_prob = net._log_prob(samples, probs)
            last_time, times["log_prob"] = time.time(), time.time() - last_time
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            
            energies = model.energy_separated(samples)
            loss_pre_info = {enr_name: e.mean().item() for enr_name, e in zip(ENERGIES_NAMES, energies)}
            energy = sum(energies[:-1])
            ener_p_sus = energies[-1]
            #else:
            #    loss_pre_info = {}
            #    energy = model.energy_(samples)
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
                
            #c_mult = 1e-6
            #_fin = 0.1/(1+c_mult)
            #beta_multipl = 0.1/(beta+c_mult) -c_fin
            if fun_beta_inv is None:
                beta_coeff = 1. - beta
            else:
                beta_coeff = fun_beta_inv(beta)
            loss = log_prob + beta * energy + beta_coeff*ener_p_sus
            loss_exact = log_prob + energy + beta_coeff*ener_p_sus
            last_time, times["loss"] = time.time(), time.time() - last_time
            loss -= loss.mean()
        loss_info = {
                    "log_q":log_prob.mean().item(),
                    "energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "loss_beta": loss.mean().item(),
                     "times": times}

        loss_info.update(loss_pre_info)
        
        return samples, loss, loss_info
    return loss


def loss_fn(net, model, beta, rerun_probs=False):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        # get probability of the samples
        times = {}
        last_time, start_t = time.time(), time.time()
        if rerun_probs:
            with torch.no_grad():
                samples, _ = net.sample(num_samples)
        else:
            samples, probs = net.sample(num_samples)
        last_time, times["sample"] = time.time(), time.time() - last_time
        if rerun_probs:
            probs = net(samples)
        log_prob = net._log_prob(samples, probs)
        last_time, times["log_prob"] = time.time(), time.time() - last_time
        energies_separ = Capabil.ENERGY_SEP in model.get_capabilities()
        with torch.no_grad():
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            if energies_separ:
                energies = model.energy_separated(samples)
                loss_pre_info = {enr_name: e.mean().item() for enr_name, e in zip(ENERGIES_NAMES, energies)}
                energy = sum(energies)
            else:
                loss_pre_info = {}
                energy = model.energy_(samples)
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            loss = log_prob + beta * energy
            loss_exact = log_prob + energy
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        last_time, times["loss"] = time.time(), time.time() - last_time

        loss_info = {"energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "times": times}

        loss_info.update(loss_pre_info)
        
        return samples, loss_reinforce, loss_info
    return loss

def loss_fn_logp(net, model, beta):
    return loss_fn(net, model, beta, rerun_probs=True)

def loss_fn_no(net, model, beta):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        # get probability of the samples
        times = {}
        last_time, start_t = time.time(), time.time()
        #with torch.no_grad():
        samples, probs = net.sample(num_samples)
        last_time, times["sample"] = time.time(), time.time() - last_time
        log_prob = net.log_prob(samples)
        last_time, times["log_prob"] = time.time(), time.time() - last_time
        with torch.no_grad():
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            energy = model.energy_(samples)
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            loss = log_prob + beta * energy
            loss_exact = log_prob + energy
        loss_reinforce = torch.mean((loss) * log_prob)
        last_time, times["loss"] = time.time(), time.time() - last_time

        loss_info = {"energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "times": times}
        
        return samples, loss_reinforce, loss_info
    return loss



def loss_comp(net, model, beta):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        # get probability of the samples
        times = {}
        last_time, start_t = time.time(), time.time()
        #with torch.no_grad():
        samples, probs = net.sample(num_samples)
        last_time, times["sample"] = time.time(), time.time() - last_time
        log_prob = net.log_prob(samples)
        last_time, times["log_prob"] = time.time(), time.time() - last_time
        with torch.no_grad():
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            energy = model.energy_(samples)
            ln1p_q = torch.log1p(-torch.exp(log_prob))
            ln1p_p = torch.log1p(-torch.exp(-energy))
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            
            loss = - (ln1p_q - beta * ln1p_p)
            loss_exact = log_prob + energy
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        last_time, times["loss"] = time.time(), time.time() - last_time

        loss_info = {"energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "times": times}
        
        return samples, loss_reinforce, loss_info
    return loss

def loss_lin_fn(net, model, beta, lin_mult=1000):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        # get probability of the samples
        times = {}
        last_time, start_t = time.time(), time.time()
        #with torch.no_grad():
        samples, probs = net.sample(num_samples)
        last_time, times["sample"] = time.time(), time.time() - last_time
        log_prob = net.log_prob(samples)
        last_time, times["log_prob"] = time.time(), time.time() - last_time
        with torch.no_grad():
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            energy = model.energy_(samples)
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            loss = log_prob + beta * energy
            max_p = torch.exp(torch.max(- beta * energy))
            loss_lin = lin_mult * torch.abs(-torch.exp(- beta * energy)/max_p + torch.exp(log_prob)/max_p)
            loss += loss_lin
            loss_exact = log_prob + energy
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        last_time, times["loss"] = time.time(), time.time() - last_time

        loss_info = {"energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "times": times}
        
        return samples, loss_reinforce, loss_info
    return loss


def make_loss_detail(net, model, softness_log=1e-6, energy_separated=False):
    def loss(num_samples, beta):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        # get probability of the samples
        extra_info = {}
        start_t = int(round(time.time() * 1000))
        #with torch.no_grad():
        samples, probs = net.sample(num_samples)
        sample_t = int(round(time.time() * 1000)) - start_t
        log_prob = net._log_prob(samples, probs)
        log_prob_t = int(round(time.time() * 1000)) - sample_t - start_t 
        with torch.no_grad():
            # get samples as trajectories
            samples = net.transform_samples(samples)
            if energy_separated:
                energies_arr = model.energy_separated(samples, softness_log)
                energy = sum(energies_arr)
                extra_info["energies_mean"] = {name: x.mean() for name,x in zip(ENERGIES_NAMES,energies_arr)}
            else:
                energy = model.energy_(samples, softness_log=softness_log)
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            loss = log_prob + beta * energy
        loss_t = int(round(time.time() * 1000)) - log_prob_t - start_t
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        
        loss_reinforce_t = int(round(time.time() * 1000)) - loss_t - start_t
    #loss_reinforce = torch.mean((loss) * log_prob)
        loss_info = {"energy": energy,
                     "loss": loss,
                     "times": {
                         "sample" : sample_t,
                         "log_prob_t" : log_prob_t,
                         "loss_t" : loss_t,
                         "loss_reinforce_t" : loss_reinforce_t
                     }}
        loss_info.update(extra_info)
        return samples, loss_reinforce, loss_info
    return loss

def loss_fn_local(net, model, beta):
    def loss(num_samples):
        """
        Calculate the loss from the samples

        This uses the samples to evaluate the free energy, or
        the Kullback-Leibler divergence of the distribution from the target one

        In the case of the SIR, the samples have to be one-hot encoded

        returns: this D_KL and a dict containing the energies and
                the losses of the samples
        """
        # get probability of the samples
        times = {}
        last_time, start_t = time.time(), time.time()
        #with torch.no_grad():
        samples, probs = net.sample(num_samples)
        last_time, times["sample"] = time.time(), time.time() - last_time
        log_prob = net.log_prob(samples)
        last_time, times["log_prob"] = time.time(), time.time() - last_time
        with torch.no_grad():
            # get samples as trajectories
            samples = net.transform_samples(samples)
            last_time, times["trans_sample"] = time.time(), time.time() - last_time
            energy = model.energy_(samples)
            last_time, times["energy"] = time.time(), time.time() - last_time
            if torch.isnan(energy).sum() > 0:
                raise ValueError("Have nan energy")
            loss = log_prob + beta * energy
            loss_exact = (log_prob + energy) * log_prob
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
        last_time, times["loss"] = time.time(), time.time() - last_time

        loss_info = {"energy": energy.detach(),
                     "loss": loss_exact.detach(),
                     "times": times}
        
        return samples, loss_reinforce, loss_info
    return loss
