import torch

def one_hot_samples(x, T, q=3):
    x_hot = torch.zeros((x.shape[0],x.shape[1],T,q), 
                        device=x.device)
    times = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(0)
    t_r = x[:, :,1].unsqueeze(-1)
    t_i = x[:, :,0].unsqueeze(-1)
    x_hot[:,:,:, 0] = (times < t_i) * (times < t_r)
    x_hot[:,:,:, 1] = ((times >= t_i) * (times < t_r)) + ((times >= t_i) * (t_i > t_r))
    #    x_hot[:,:,:, 1] = ((times >= t_i) * (times < t_r)) * (t_i < t_r) + ((times >= t_i) * (t_i > t_r))
    x_hot[:,:,:, 2] = ((times >= t_r) * (t_r >= t_i)) + ((times >= t_r) * (times < t_i))
    #x_hot[:,:,:, 2] = (times >= t_r) * (t_r >= t_i) + (times >= t_r) * (times < t_i) * (t_r < t_i) 
    assert(torch.all(x_hot.sum(dim=3)==1))
    return x_hot

def one_hot_samples_relative_r(x, T, q=3):
    x_hot = torch.zeros((x.shape[0],x.shape[1],T,q), 
                        device=x.device)
    times = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(0)
    t_r = x[:, :,1].unsqueeze(-1)
    t_i = x[:, :,0].unsqueeze(-1)
    x_hot[:,:,:, 0] = (times < t_i)
    x_hot[:,:,:, 1] = ((times >= t_i) * (times < t_i + t_r))
    x_hot[:,:,:, 2] = (times >= t_i + t_r)
    return x_hot

def one_hot_conf_from_times(x, T):
    """
    Return one hot configuration of the epidemy,
    from infection and recovery times.
    It is implied that the times indicate the exact instant in which
    the nodes become I and R

    Same results of method above, just 12% faster
    """
    q = 3
    x_hot = torch.zeros((x.shape[0],x.shape[1],T,q),
                        device=x.device)
    times = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(0)
    t_r = x[:, :,1].unsqueeze(-1)
    t_i = x[:, :,0].unsqueeze(-1)

    ti_lesseq_tr = t_i <= t_r
    ti_bigger_tr = torch.logical_not(ti_lesseq_tr)
    tim_less_tr = (times < t_r)
    tim_greq_ti = (times >= t_i)
    ### DESCRIPTION WITHOUT PRESAVING
    ###x_hot[:,:,:, 0] = (times < t_i) * (times < t_r)
    ###x_hot[:,:,:, 1] = ((times >= t_i) * (times < t_r)) * (t_i <= t_r) + ((times >= t_i) * (t_i > t_r))
    ###x_hot[:,:,:, 2] = (times >= t_r) * (t_i <= t_r) + (times >= t_r) * (times < t_i) * (t_i > t_r)

    x_hot[:,:,:, 0] = torch.logical_not(tim_greq_ti) * tim_less_tr
    x_hot[:,:,:, 1] = tim_greq_ti * ( ti_lesseq_tr * tim_less_tr + ti_bigger_tr)
    x_hot[:,:,:, 2] = torch.logical_not(tim_less_tr) * (ti_lesseq_tr + ti_bigger_tr * torch.logical_not(tim_greq_ti))
    return x_hot
