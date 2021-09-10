from learn.train import fill_res, make_training_step, create_res_dict, save_results_df
from learn.opt import make_opt
import learn.l_utils as learn_utils
import torch
import numpy as np
import pandas as pd
import time

def print_step(res,
               name_file="log",
               print_log=True,
               max_num_print=None):
    
    src_string = learn_utils.format_src_print(res["sources"][-1][:max_num_print])

    m_grad = res["max_grad"][-1]
    m_grad_print = "{:.5f}".format(m_grad) if m_grad > 0 else "0.0"

    loss=res["loss"][-1]
    step=res["step"][-1]
    I=res["I"][-1]
    R=res["R"][-1]
    ener=res["energy"][-1]
    source=src_string
    beta=res["beta"][-1]
    std=res["loss_std"][-1]
    max_grad=m_grad_print
    T_obs =res["t_obs"][-1]
    zero_pw=res["num_zero_pw"][-1]
    mu=res["mu"][-1]
    times_out = ", ".join(["{0}: {1:d}".format(k, int(l[-1]*1000)) for k, l in res["times"].items()])
    params=""
    if "lamb" in res.keys():
        lam=res["lamb"][-1]
        params += f"lam:{lam:.3}"
    if "gamma1" in res.keys():
        gam1=res["gamma1"][-1]
        gam2=res["gamma2"][-1]
        params+= f" gamma1:{gam1:.3}, gamma2:{gam2:.3}"
    tot_time=int(sum([l[-1]*1000 for k, l in res["times"].items()]))
    print(f"\r{step:4} beta: {beta:5.4f}, {params}, mu:{mu:.3}, loss: {loss:02.3f}, std: {std:02.3f}  ener: {ener:.5f}, max_grad = {max_grad}, count_zero_pw = {zero_pw:.3f}, num_I = {I:.3f}, num_R = {R:.4f} (T_obs={T_obs}) -- took {tot_time:6d} ms: {times_out} -- source: {source}", end="    ")
    print(" ", end="")
    #print("params: ",*["{0}: {1}".format(k, int(l[-1]*1000)) for k, l in res["times"].items()], end="    ")


def opt_param_init(model, param_init=0.1, name="lambda", dtype=torch.float, device="cpu", lr=1e-3):
    device_torch = torch.device(device)
    param_learn = torch.tensor(param_init, dtype=dtype, device=device_torch, requires_grad=True)
    opt_param = make_opt([param_learn], lr=lr, betas=(0.5, 0.9))
    model.extra_params[name]=param_learn.detach().clone()
    return param_learn, opt_param

def logP(model, x):
    logP = model.energy(x)
    return logP.sum()

def logP_logQ(model, x, log_q_x, baseline_norm=-200):
    #print(log_q_x, model.energy(x))
    energy=model.energy(x)
    logP = - energy - log_q_x
    model.extra_params["max_Z"]=max(torch.max(logP.detach()), model.extra_params["max_Z"])
    logP-=model.extra_params["max_Z"]
    #print(max(logP), min(logP), min(model.energy(x)))
    return -torch.exp(logP).sum()


def learn_lamb_mu(model, samples, params, opt_params, log_q_x=0, eps=1e-6, clip_val=1):
    lamb_param, mu_param = params
    lamb_opt, mu_opt = opt_params
    lamb_opt.zero_grad()
    mu_opt.zero_grad()
    for i in range(model.N):
        model.logp_lam[i][model.logp_lam[i]!=0] = torch.log1p(-lamb_param)
    model.mu = mu_param
    logP_ = logP(model, samples)
    logP_.backward()
    torch.nn.utils.clip_grad_value_(params, clip_val)
    lamb_opt.step()
    mu_opt.step()
    with torch.no_grad():
        lamb_param.clamp_(eps, 1-eps)    
        mu_param.clamp_(eps, 1-eps)      
        for i in range(model.N):
            model.logp_lam[i][model.logp_lam[i]!=0] = torch.log1p(-lamb_param.detach().clone())
        model.mu=mu_param.detach().clone()
        model.extra_params["mu"]=mu_param.detach().clone()
        model.extra_params["lamb"]=lamb_param.detach().clone()
        for i in range(model.N):
            model.logp_lam[i].detach_()
        model.mu.detach_() 
        
def learn_lamb_mu_q(model, samples, params, opt_params, log_q_x, eps=1e-6, clip_val=1):
    lamb_param, mu_param = params
    lamb_opt, mu_opt = opt_params
    lamb_opt.zero_grad()
    mu_opt.zero_grad()
    for i in range(model.N):
        model.logp_lam[i][model.logp_lam[i]!=0] = torch.log1p(-lamb_param)
    model.mu = mu_param
    logP_ = logP_logQ(model, samples, log_q_x)
    logP_.backward()
    torch.nn.utils.clip_grad_value_(params, clip_val)
    lamb_opt.step()
    mu_opt.step()
    with torch.no_grad():
        lamb_param.clamp_(eps, 1-eps)    
        mu_param.clamp_(eps, 1-eps)      
        for i in range(model.N):
            model.logp_lam[i][model.logp_lam[i]!=0] = torch.log1p(-lamb_param.detach().clone())
        model.mu=mu_param.detach().clone()
        model.parameters["mu"]=mu_param.detach().clone()
        model.parameters["lamb"]=lamb_param.detach().clone()
        for i in range(model.N):
            model.logp_lam[i].detach_()
        model.mu.detach_() 

        
def learn_gamma_mu(model, samples, params, opt_params, q_x=0, eps=1e-6, clip_val=0.01):
    gamma_param, mu_param = params
    gamma_opt, mu_opt = opt_params
    gamma_opt.zero_grad()
    mu_opt.zero_grad()
    for i in range(model.N):
        model.logp_lam[i] = - gamma_param * model.deltas[i]
    model.mu = mu_param
    logP_ = logP(model, samples)
    logP_.backward()
    torch.nn.utils.clip_grad_norm_(params, clip_val)
    gamma_opt.step()
    mu_opt.step()
    #print(f"gamma {gamma_param}")
    with torch.no_grad():
        gamma_param.clamp_(eps, 1/eps)    
        mu_param.clamp_(eps, 1-eps)      
        gamma=gamma_param.detach().clone()
        for i in range(model.N):
            model.logp_lam[i] = - gamma * model.deltas[i]
        
        model.mu=mu_param.detach().clone()
        model.extra_params["mu"]=mu_param.detach().clone()
        model.extra_params["lamb"]=gamma_param.detach().clone()
        for i in range(model.N):
            model.logp_lam[i].detach_()
        model.mu.detach_() 
        
def learn_two_gamma(model, samples, params, opt_params, q_x=0, eps=1e-6, clip_val=0.01):
    gamma1_param, gamma2_param = params
    gamma1_opt, gamma2_opt = opt_params
    gamma1_opt.zero_grad()
    gamma2_opt.zero_grad()
    for i in range(model.N):
        if i in model.nodes_1:
            model.logp_lam[i] = - gamma1_param * model.deltas[i]
        else:
            model.logp_lam[i] = - gamma2_param * model.deltas[i]

    logP_ = logP(model, samples)
    logP_.backward()
    torch.nn.utils.clip_grad_norm_(params, clip_val)
    gamma1_opt.step()
    gamma2_opt.step()
    #print(f"gamma {params}")
    with torch.no_grad():
        gamma1_param.clamp_(eps, 1/eps)    
        gamma2_param.clamp_(eps, 1/eps)    
        gamma1=gamma1_param.detach().clone()
        gamma2=gamma2_param.detach().clone()
        for i in range(model.N):
            if i in model.nodes_1:
                model.logp_lam[i] = - gamma1* model.deltas[i]
            else:
                model.logp_lam[i] = - gamma2 * model.deltas[i]
        model.extra_params["gamma1"]=gamma1
        model.extra_params["gamma2"]=gamma2
        for i in range(model.N):
            model.logp_lam[i].detach_()

        
def train_beta_params(net,
                     optimizer,
                     model,
                     name_file,
                     loss_fn,
                     T_obs,
                     params,
                     opt_params,
                     learn_params = learn_lamb_mu,
                     results = None,
                     train_step = make_training_step,
                     betas=np.arange(0, 1+1e-3, 1e-3),
                     num_samples = 10000,
                     print_log=True,
                     max_num_print=100,
                     save_every = 10,
                     clip_grad = None
                     ):

    '''# "touch" file
    if print_log:
        with open(name_file + ".log", "w") as f:
            f.write("")'''
    if results is None:
        results = create_res_dict(model)
        last_step = 0
    else:
        last_step = results["step"][-1] +1

    try:
        for i_b, beta in enumerate(betas):
              
            loss = loss_fn(net, model, beta)
            samples, loss_info = train_step(
                num_samples, loss, optimizer, clip_grad = clip_grad, net = net)
            start_time = time.time()
            max_grad = 0
            learn_params(model, samples, params, opt_params, loss_info["log_prob_q"])
            M = net.marginals_(samples)
            results = fill_res(i_b+last_step, beta, T_obs, model, M, loss_info, max_grad, results)
            took_time = time.time() - start_time
            results["times"]["stats"] = (took_time,)
            print_step(results, max_num_print=max_num_print)
            #results
            if i_b != 0 and i_b % save_every == 0:
                save_results_df(results, M, name_file)
    
        ## Finished training
        torch.save(net, name_file+".pt")

        save_results_df(results, M, name_file)
        print()
    except KeyboardInterrupt as interr:
        print("\nCaught KeyboardInterrupt, terminating...")
        del samples
        del loss_info
        del results["times"]

        raise InterruptedError from interr

    del samples
    del loss_info
    del results["times"]
    return results
