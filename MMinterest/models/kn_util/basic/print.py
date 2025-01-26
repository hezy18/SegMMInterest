def dict2str(x, delim=": ", sep="\n", fmt=".2f"):
    """
    Convert a dictionary to a string

    Parameters
    ----------
    x : dict
        Dictionary to be converted to a string

    Returns
    -------
    str
        String representation of the dictionary
    """
    kv_list = []
    for k, v in x.items():
        sv = "{:{}}".format(v, fmt)
        kv_list.append("{k}{delim}{v}".format(k=k, delim=delim, v=sv))

    return sep.join(kv_list)

def max_memory_allocated():
    """
    Get the maximum memory allocated by pytorch

    Returns
    -------
    int
        Maximum memory allocated by pytorch
    """
    import torch
    return torch.cuda.max_memory_allocated() / 1024.0 / 1024.0