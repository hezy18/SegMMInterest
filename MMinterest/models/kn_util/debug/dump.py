from ..basic.file import save_pickle

def load_context_to_pickle(local_dict, keys, to_dir="./"):
    for k in keys:
        variable = local_dict[k]
        save_pickle(variable, f"{k}.pkl")