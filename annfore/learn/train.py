import time
import torch
import numpy as np
import pandas as pd
from .l_utils import sort_I
from torch.nn.utils.clip_grad import clip_grad_norm_
import threading, queue

# don't call the imported model with an already used name
# as it messes up the namespace
from ..learn import l_utils as learn_utils
from ..learn.losses import ENERGIES_NAMES

from ..utils.logging import format_time


def create_res_dict(model):
    results = {}
    data_results = ["step", "beta", "energy", "std_energy", 
                    "loss", "loss_std", "S", "I", "R", "t_obs", 
                    "num_source", "sources", "max_grad", "num_zero_pw"]
    
    for param in model.params():
        data_results.append(param)

    for res in data_results:
        results[res] = []
    return results

EXTRA_LOSS_KEYS = ENERGIES_NAMES + ["log_q", "loss_beta"]# + ["loss_mean","loss_var"]

def fill_res(step, beta, t_obs, model, margs, loss_info, max_grad, results):
    allpars =  model.params()
    for param in allpars:
        if allpars[param] is None:
            results[param].append(None)
        else:
            results[param].append(float(allpars[param]))
    results["step"].append(step)
    results["beta"].append(beta)
    results["max_grad"].append(max_grad)
    results["energy"].append(loss_info["energy"].mean().cpu().item())
    results["std_energy"].append(loss_info["energy"].std().cpu().item())
    results["loss"].append(float(loss_info["loss"].mean()))
    results["loss_std"].append(float(loss_info["loss"].std()))
    results["S"].append(float(margs[:, t_obs, 0].sum()))
    results["I"].append(float(margs[:, t_obs, 1].sum()))
    if margs.shape[2] > 2:
        results["R"].append(float(margs[:, t_obs, 2].sum()))
    else:
        results["R"].append(0)
    results["t_obs"].append(t_obs)
    results["num_zero_pw"].append(model.count_zero)
    results["num_source"].append(float(margs[:, 0, 1].sum()))
    sources = sort_I(margs)
    results["sources"].append(sources[0:20].cpu().numpy())
    for time in loss_info["times"]:
        if "times" not in results:
            results["times"] = {}
        if time not in results["times"]:
            results["times"][time] = []
        results["times"][time].append(loss_info["times"][time])
    
    for k in EXTRA_LOSS_KEYS:
        if k in loss_info.keys():
            if k not in results.keys():
                results[k] = []
            results[k].append(loss_info[k])
    
    return results



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
    ener_str=""
    logq_str=""
    for ener_k in EXTRA_LOSS_KEYS:
        if ener_k in res.keys():
            ener_str += f"{ener_k}:{res[ener_k][-1]:.2f} "
    else:
        ener_str+=f"ener: {res['energy'][-1]:.2f}"
    source=src_string
    beta=res["beta"][-1]
    std=res["loss_std"][-1]
    max_grad=m_grad_print
    T_obs =res["t_obs"][-1]
    zero_pw=res["num_zero_pw"][-1]
    times_out = ", ".join(["{0}: {1:d}".format(k, int(l[-1]*1000)) for k, l in res["times"].items()])
    tot_time=int(sum([l[-1]*1000 for k, l in res["times"].items()]))
    print(f"\r{step:4} beta: {beta:5.4f}, loss: {loss:02.3f}, std: {std:02.3f}, {ener_str}, max_grad = {max_grad}, count_zero_pw = {zero_pw:.2f}, num_I = {I:.3f}, num_R = {R:.4f} (T_obs={T_obs}) -- took {tot_time:6d} ms: {times_out} -- source: {source}", end="    ")
    #print(" ", end="")
    return tot_time
    #print("params: ",*["{0}: {1}".format(k, int(l[-1]*1000)) for k, l in res["times"].items()], end="    ")

def print_remaining(i_b, max_b, cache):
    avg_t_secs = np.nanmean(cache)/1000
    rem_step = max_b - i_b -1
    rem_time = rem_step * avg_t_secs
    rem_time_str = format_time(rem_time, compact=True)
    print("- remain time: "+ rem_time_str, end="")

def make_training_step(num_samples, loss, optimizer, net=None, clip_grad = None):
    """
    Calculate the loss and backpropagate, step the optimizer
    """
    optimizer.zero_grad()

    samples_good, loss_reinforce, loss_info = loss(num_samples)
    start_t = time.time()
    loss_reinforce.backward()
    backward_t = time.time()
    loss_info["times"]["backward"]  = backward_t  - start_t
    if clip_grad != None:
        #print("clipped")
        clip_grad_norm_(optimizer.param_groups[0]["params"], clip_grad)
    optimizer.step()
    loss_info["times"]["optim_step"]  = time.time() - backward_t
    return samples_good, loss_info

def make_train_single(i, loss_coeff, samples, net, optimizer):
    torch.cuda.empty_cache()
    #GPUtil.showUtilization()
    #GPUtil.showUtilization()
    loss_reinforce = torch.mean(loss_coeff * net.log_prob_i(i, samples))
    loss_reinforce.backward()
    del loss_reinforce
    #GPUtil.showUtilization()
    optimizer[i].step()
    optimizer[i].zero_grad()
    #if clip_grad != None:
        #print("clipped")
    #    clip_grad_norm_(optimizer[i].param_groups[0]["params"], clip_grad)
            


def make_training_step_local(num_samples, loss, optimizer, net, clip_grad = None, verbose=False):
    """
    Calculate the loss and backpropagate, step the optimizer
    """
    
    samples, loss_coeff, loss_info = loss(num_samples)
    if samples.dtype == torch.long or samples.dtype == torch.int:
        dtype = torch.float32
    else:
        dtype = samples.dtype
    log_prob_sample=torch.zeros(num_samples, device=samples.device, dtype=dtype)
    start_t = time.time()
    for i in range(net.N):
        if verbose:
            print(f" {i}", end=" ")
        if not( optimizer[i] == [] or optimizer[i] is None):
            optimizer[i].zero_grad()
            log_prob_i = net.log_prob_i(i, samples)
            loss_reinforce = torch.mean(loss_coeff * log_prob_i)
            log_prob_sample += log_prob_i.detach().clone()
            loss_reinforce.backward()
            optimizer[i].step()
            if clip_grad is not None:
                #print("clipped")
                clip_grad_norm_(optimizer[i].param_groups[0]["params"], clip_grad)
    
    loss_info["times"]["optim_step"]  = time.time() - start_t
    loss_info["log_prob_q"]=log_prob_sample
    return samples, loss_info


def make_training_step_local_par(num_samples, loss, optimizer, net=None, clip_grad = None, verbose=False):
    """
    Calculate the loss and backpropagate, step the optimizer
    """
    
    if net == None:
        print("ERROR make_training_step_local need net")
    N = net.N
    
    samples, loss_coeff, loss_info = loss(num_samples)

    start_t = time.time()
    q = queue.Queue()
    
    def step_func(q):
        while not q.empty():
            i=q.get()
            if verbose:
                print(f" {i}", end=" ")
            if not( optimizer[i] == [] or optimizer[i] is None):
                optimizer[i].zero_grad()
                loss_reinforce = torch.mean(loss_coeff * net.log_prob_i(i, samples))
                loss_reinforce.backward()
                optimizer[i].step()
                if clip_grad != None:
                    #print("clipped")
                    clip_grad_norm_(optimizer[i].param_groups[0]["params"], clip_grad)
            q.task_done()
        
    for item in range(N):
        q.put(item)
    
    for item in range(N):
        worker = threading.Thread(target=step_func, args=(q,))
        worker.start()
    #print("waiting for queue to complete", q.qsize(), "tasks")
            # block until all tasks are done
    q.join()
    #print('All work completed')
    
    loss_info["times"]["optim_step"]  = time.time() - start_t
    return samples, loss_info



def save_results_df(results, marginals, name_file):
    """
    Save marginals and dictionary
    """
    margs_out = marginals.cpu().numpy()
    results_pd = results.copy()
    print(results_pd.keys())
    try:
        del results_pd["times"]
    except:
        pass
    psrc_hist = np.stack(results_pd["sources"])
    del results_pd["sources"]
    results_pd = pd.DataFrame.from_dict(results_pd, orient='index')
    
    results_pd = results_pd.transpose()
    
    results_pd.to_csv(name_file+".gz")
    #pd.DataFrame(margs_out).to_csv(name_file+"_M.gz")
    np.savez_compressed(name_file+"_margs.npz",marginals=margs_out, sources=psrc_hist)

# TODO: Write a new method for the SIR General case

T_CACHE=15
def train_beta(net,
                     optimizer,
                     model,
                     name_file,
                     loss_fn,
                     T_obs,
                     results = None,
                     train_step = make_training_step,
                     betas=np.arange(0, 1+1e-3, 1e-3),
                     num_samples = 10000,
                     print_log=True,
                     max_num_print=100,
                     save_every = 10,
                     clip_grad = None,
                     save_net=False,
                     loss_extra_args:dict=None
                     ):

    '''# "touch" file
    if print_log:
        with open(name_file + ".log", "w") as f:
            f.write("")'''
    last_step = 0
    if results is None:
        results = create_res_dict(model)
    else:
        if len(results["step"]):
            last_step = results["step"][-1]+1

    if loss_extra_args is None:
        loss_extra_args = dict()
    else:
        print(loss_extra_args)

    try:
        max_betas = len(betas)
        time_cache = np.full(T_CACHE, np.nan)
        i_cache = 0
        for i_b, beta in enumerate(betas):
              
            loss = loss_fn(net, model, beta, **loss_extra_args)
            samples, loss_info = train_step(
                num_samples, loss, optimizer, clip_grad = clip_grad, net = net)
            start_time = time.time()
            max_grad = 0
            """for p in net.parameters():
                if p.grad != None:
                    max_grad = max(abs(p.grad.norm()), max_grad)
            """
            M = net.marginals_(samples)
            results = fill_res(i_b+last_step, beta, T_obs, model, M, loss_info, max_grad, results)
            took_time = time.time() - start_time

            results["times"]["stats"] = (took_time,)
            time_tot = print_step(results, max_num_print=max_num_print)
            time_cache[i_cache] = time_tot
            i_cache = (i_cache+1) % T_CACHE
            print_remaining(i_b, max_betas, time_cache)
            #results
            if i_b != 0 and i_b % save_every == 0:
                save_results_df(results, M, name_file)
    
        ## Finished training
        if save_net:
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

def train_saving_marginals(net,
                     optimizer,
                     model,
                     name_file,
                     loss_fn,
                     T_obs,
                     train_step = make_training_step,
                     results = None,
                     num_iter = 100,
                     num_samples = 10000,
                     print_log=True,
                     max_num_print=100,
                       clip_grad = None
                     ):

    # "touch" file
    '''if print_log:
        with open(name_file + ".log", "w") as f:
            f.write("")'''
    if results is None:
        results = create_res_dict(model)
        last_step = 0
    else:
        last_step = results["step"][-1] +1
    M = net.marginals()
    print("starting marginals")
    betas = [1]*(num_iter-1)
    for i_b, beta in enumerate(betas):
        
        
        loss = loss_fn(net, model, beta)
        samples, loss_info = train_step(
            num_samples, loss, optimizer, clip_grad = clip_grad, net = net)
        start_time = time.time()
        max_grad = 0
        """for p in net.parameters():
            if p.grad != None:
                max_grad = max(abs(p.grad.norm()), max_grad)
        """
        M += net.marginals_(samples)
        results = fill_res(i_b+last_step, beta, T_obs, model, M/(i_b+2), loss_info, max_grad, results)
        took_time = time.time() - start_time
        results["times"]["stats"] = (took_time,)
        print_step(results, max_num_print=max_num_print)
        #results
                
    M /= num_iter
    save_results_df(results, M, name_file)
    print()
    
    return results

def train_beta_changing(net,
                     optimizer,
                     model,
                     name_file,
                     loss_fn,
                     T_obs,
                     betas,
                     par_changes = None,
                     results = None,
                     num_samples = 10000,
                     print_log=True,
                     max_num_print=100,
                     save_every = 100,
                       clip_grad = None
                     ):
    """
    par_changes has to be list of tuples:
        (function_to_apply, params, fixed_params)
    
    params should be 2D with shape (num_steps, n_pars)
    """

    # "touch" file
    if print_log:
        with open(name_file + ".log", "w") as f:
            f.write("")
    if results is None:
        results = create_res_dict(model)
        last_step = 0
    else:
        last_step = results["step"][-1] +1
    
    num_betas = len(betas)
    changing = par_changes is not None

    for i_b, beta in enumerate(betas):
        if changing:
            for changes in par_changes:
                changes[0](*changes[1][i_b], *changes[2])
        
        loss = loss_fn(net, model, beta)
        samples, loss_info = make_training_step(
            num_samples, loss, optimizer, clip_grad)
        start_time = time.time()
        max_grad = 0

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
    
    return results

def train_model(net,
                     optimizer,
                     model,
                     name_file,
                     loss_fn,
                     T_obs,
                     results = None,
                     p_sources=np.exp(np.linspace(-4, -15, 2000)),
                     p_ws=np.exp(np.linspace(-1, -6, 100)),
                     num_samples = 10000,
                     divide_std=False,
                     print_log=True,
                     max_num_print=100,
                     save_every = 10
                     ):

    # "touch" file
    if print_log:
        with open(name_file + ".log", "w") as f:
            f.write("")
    if not results:
        results = create_res_dict(model)
    
    beta = 1
    model.p_w = p_ws[0]
    model.p_source = p_sources[0]
    for i_p_w, p_w in enumerate(p_ws):
        model.p_w = p_w
        loss = loss_fn(net, model, beta)
        samples, loss_info = make_training_step(
            num_samples, loss, optimizer)
        
        M = net.marginals_(samples)
        fill_res(i_p_w, beta, T_obs, model, M, loss_info,0., results)
        print_step(results, max_num_print=max_num_print)
        #results
        if i_p_w % save_every == 0:
            results_pd = results.copy()
            del results_pd["times"]
            results_pd = pd.DataFrame(results_pd)
            results_pd.to_csv(name_file+".gz")
            pd.DataFrame(M).to_csv(name_file+"_M.gz")
    
    for i_p_source, p_source in enumerate(p_sources):
        model.p_source = p_source
        loss = loss_fn(net, model, beta)
        samples, loss_info = make_training_step(
            num_samples, loss, optimizer)
        
        M = net.marginals_(samples)
        max_grad = 0
        results = fill_res(i_p_source, beta, T_obs, model, M, loss_info, max_grad, results)
        print_step(results, max_num_print=max_num_print)
        #results
        if i_p_source % save_every == 0:
            results_pd = results.copy()
            del results_pd["times"]
            results_pd = pd.DataFrame(results_pd)
            results_pd.to_csv(name_file+".gz")
            pd.DataFrame(M).to_csv(name_file+"_M.gz")
    ## Finished training
    torch.save(net, name_file+".pt")
    
    results_pd = results.copy()
    del results_pd["times"]
    results_pd = pd.DataFrame(results_pd)
    results_pd.to_csv(name_file+".gz")
    pd.DataFrame(M).to_csv(name_file+"_M.gz")
    
    
    return results
