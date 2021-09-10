import itertools
import numpy as np

def get_next_n_neighs(dict_neighs):
    """
    Given the neighbors in a dict, return
    both the neihbors and the next nearest neighbors
    """
    g_next_n = [set() for i in range(len(dict_neighs))]
    for i, _ in enumerate(g_next_n):
        g_next_n[i].update(dict_neighs[i])
        for j in dict_neighs[i]:
            g_next_n[i].update(dict_neighs[j])
        
        if i in g_next_n[i]:
            g_next_n[i].remove(i)

    return g_next_n

def add_neighs(neighs_current, orig_neighs):
    """
    Given the current neighbors and the orginal neighbors in dict `orig_neighs`,
    add the neighbors of the current neighbors
    """
    out_dict = {}
    for i, neighs in neighs_current.items():
        out_dict[i] = set(neighs)
        for j in neighs:
            if j in orig_neighs.keys() and j != i:
                nn = set(orig_neighs[j])
                if i in nn:
                    nn.remove(i)
                out_dict[i].update(nn)
    return out_dict

def filter_neighs_order(neighs):
    """
    Filter the neighbors, returning a dict
    with only the ones with lower index
    """
    n_out = {}
    for i in range(len(neighs)):
        n_out[i] = set(x for x in neighs[i] if x < i)
    return n_out

def find_neighs(contacts, N = None,
                         only_minor = False,
                        next_near_neigh=False):
    """
    Find the neighbor nodes, with optionally the next
    nearest neighbors and only with lower index than the
    considered node

    if `next_near_neighs` is an integer, outputs the
    the `next_near_neighs`+1 nearest neighbors
    
    Useful for the Sir Path
    """
    if not isinstance(contacts, np.ndarray):
        contacts = np.array(contacts)
    if N == None:
        N = int(np.max(contacts[:,1:3])) + 1
    #connections = model.lamdas.nonzero()
    d_neighs = {i:set() for i in range(N)}

    for cont in contacts:
        i = int(cont[1])
        j = int(cont[2])
        d_neighs[j].add(i)
        # g.add_edge(j,i)

    curr_neighs = d_neighs
    if next_near_neigh and next_near_neigh > 0:
        for _ in range(next_near_neigh):
            curr_neighs = add_neighs(curr_neighs, d_neighs)

    if only_minor:
        neighs_dict = filter_neighs_order(curr_neighs)
    else:
        neighs_dict = curr_neighs

    return [tuple(neighs_dict[k]) for k in sorted(neighs_dict.keys())]

def create_neighs_basedict(neighs_dict, N, T, nearest_times=None):
    out_dict = dict()
    for i in range(N):
        neighs = neighs_dict[i]
        all_before = list(itertools.product(range(T), neighs))
        listprev = []

        for t in range(T):
            current = (t, i)
            out_dict[current] = list(listprev)
            if nearest_times is not None:
                if not isinstance(nearest_times,int):
                    raise ValueError("Nearest time parameters has to be int")
                st_t = max(t-nearest_times, 0)
                end_t = min(t+nearest_times+1, T)
                t_range = range(st_t,end_t)
                out_dict[current].extend(itertools.product(t_range, neighs))
            else:
                out_dict[current].extend(all_before)
            listprev.append(current)
    return out_dict

def find_neighs_time_order(contacts,
                        next_near_neigh=False,
                        nearest_times=None):
    """
    Find the correct neighbors of each individual,
    that have lower index value and are either a nearest neighbor
    or next nearest neighbors

    Contacts have to be symmetric in order to get symmetric neighbors

    with nearest_times == None, all the times of the selected 
    individuals are considered
    """
    
    T = int(max(contacts[:, 0]) + 2)
    N = int(max(contacts[:, 1]) + 1)

    if not (isinstance(nearest_times,int) or nearest_times is None):
        raise ValueError("Input integer value for the nearest times")

    d_neighs = {i:set() for i in range(N)}
    for cont in contacts:
        i = int(cont[1])
        j = int(cont[2])
        d_neighs[j].add(i)
    
    if isinstance(next_near_neigh, bool):
        if(next_near_neigh):
            neighs_dict = get_next_n_neighs(d_neighs)
        else:
            neighs_dict = d_neighs
    elif isinstance(next_near_neigh, int):
        neighs_dict = d_neighs
        if next_near_neigh > 0:
            for _ in range(next_near_neigh):
                neighs_dict = add_neighs(neighs_dict, d_neighs)
        else:
            print("next_near_neigh <= 0, not looking for next neighbors")
    else:
        raise ValueError("""Input either a bool (True, False)
         or an integer for the next nearest neighbors""")

    for i in range(len(neighs_dict)):
        neighs_dict[i] = set(x for x in neighs_dict[i] if x < i)

    return create_neighs_basedict(neighs_dict, N, T, nearest_times)


def find_neighs_time_grouped(neighs_graph, T):
    """
    Compute the neighbors and the indices for the
    case of grouped SI.
    Plug straight onto nn_si and it should work
    """

    iter_prod = itertools.product

    vals = [np.array([list(iter_prod(range(T), [k])) for k in list(v,)],dtype=int) for i,v in enumerate(neighs_graph)]

    vals_neighs = [v.reshape(np.prod(v.shape[:-1]), -1) if len(v) > 0 else v for v in vals]

    neighs_not_full = {tuple(iter_prod(range(T),[i])):  tuple([tuple(ll) for ll in vals_neighs[i]]) for i,v in enumerate(neighs_graph)}

    return neighs_not_full