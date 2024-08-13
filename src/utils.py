
import pickle 
from numba.typed import Dict, List
from numba.core import types
import numba
import numpy as np
from itertools import islice
from pathlib import Path
import warnings
import os 


def load_data(data_dir, 
              year, 
              connected_node_file = None, 
              layer_types: list = ["neighbor", "colleague"],
              sample_size: int = -1
              ):
    """Load layered network data
    
    Args:
        data_dir (str): path to the directory with the layers
        year (int): year of the data to use
        connected_node_file (str): name of the file with nodes in the biggest connected component. 
            Needs to be stored as "`data_dir`/`connected_node_file`_`year`.pkl".
            If not supported, it will take the nodes from the family network as the connected set. 
            This should only be used on fake data.
        layer (list, optional): layers of data to load. Must be a subset of 
            ["family", "colleague", "classmate", "neighbor", "household"]
        sample_size (int, optional): If non-negative, returns a random sample of this size of connected nodes.

    Returns:
        tuple: (
            list of users, 
            list of layers, 
            dictionary of users indicating on which layers they have at least one connection
            )
    
    Raises:
        UserWarning when `connected_node_file` is not provided.
    
    """

    possible_layers = ["family", "colleague", "classmate", "neighbor", "household"]
    assert all([layer in possible_layers for layer in layer_types])

    if connected_node_file:
        with Path(data_dir + connected_node_file + "_" + str(year) + ".pkl").open("rb") as pkl_file:
            unique_users = list(pickle.load(pkl_file))
    else:
        warnings.warn("connected_node_file not provided; using edges from family network. Do this only with fake data.")
        layer_types = list(set(layer_types + ["family"]))


    layers = []
    for ltype in layer_types:
        with Path(data_dir + ltype + "_" + str(year) + "_adjacency_dict.pkl").open("rb") as pkl_file:
            edges = dict(pickle.load(pkl_file))

            if not connected_node_file and ltype == "family":
                unique_users = list(edges.keys())

            layers.append(edges)

    node_layer_dict = {}
    for user in unique_users:
        node_layer_dict[user] = []
        
        for i, layer in enumerate(layers):
            if user in layer:
                if len(layer[user]) > 0:
                    node_layer_dict[user].append(i)


    if sample_size > 0:
        rng = np.random.default_rng(seed=95359385252)
        unique_users = list(rng.choice(unique_users, size=sample_size))

    return unique_users, layers, node_layer_dict



def convert_to_numba(users: list, layers: list, node_layer_dict: dict):
    """Convert python data structures to numba-compatible ones.
    
    Args:
        users: list of node identifiers.
        layers: list of adjacency lists
        node_layer_dict: dictionary indicating 

    Returns:
        the same objects with data types compatible for numba acceleration.
    
    """

    users_numba = List(users)
    users_numba = numba.int64(users_numba)

    node_layer_dict_numba = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )   
    for k, v in node_layer_dict.items():
        k = types.int64(k)
        node_layer_dict_numba[k] = np.asarray(v, dtype=np.int64)

    layers_numba = List()
    for layer in layers: 
        layer_numba = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )
        for k, v in layer.items():
            k = types.int64(k)
            layer_numba[k] = np.asarray(v, dtype=np.int64)
    
        layers_numba.append(layer_numba)

    return users_numba, layers_numba, node_layer_dict_numba


# https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def get_n_cores(interactive: bool=False):
    """Returns the number of cores to use. 

    Args:
        interactive (bool, optional): if true, returns *half* of the total cores available.     
    """
    cpus_avail = os.sched_getaffinity(0)
    # print(f"Have the following CPU cores: {cpus_avail}") 
    n_cores = len(cpus_avail)
    if interactive:
        n_cores = n_cores // 2
    return n_cores
