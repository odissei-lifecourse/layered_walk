
import pickle 
from numba.typed import Dict, List
from numba.core import types
import numpy as np
from itertools import islice, zip_longest
from pathlib import Path


def load_data(data_dir, layer_types: list = ["neighbor", "colleague"]):
    """Load layered network data
    
    Args:
        layer (list): layers of data to load. Must be a subset of ["family", "colleague", "education", "neighbor", "household"]

    Returns:
        tuple: (
            list of users, 
            list of layers, 
            dictionary of users indicating on which layers they have at least one connection
            )
    
    """
    possible_layers = ["family", "colleague", "education", "neighbor", "household"]
    assert all([layer in possible_layers for layer in layer_types])
    layers = []
    for ltype in layer_types:
        with Path(data_dir + ltype + "_adjacency_dict.pkl").open("rb") as pkl_file:
            edges = dict(pickle.load(pkl_file))
            # edges_keep = dict((u, edges[u]) for u in users)
            layers.append(edges)

    users = list(layers[0].keys())


    node_layer_dict = {}
    for user in users:
        node_layer_dict[user] = []
        
        for i, layer in enumerate(layers):
            if user in layer:
                if len(layer[user]) > 0:
                    node_layer_dict[user].append(i)

    return users, layers, node_layer_dict



def convert_to_numba(users: list, layers: list, node_layer_dict: dict, ):
    """Convert python data structures to numba-compatible ones.
    
    Args:
        users: list of node identifiers.
        layers: list of adjacency lists
        node_layer_dict: dictionary indicating 

    Returns:
        the same objects with data types compatible for numba acceleration.
    
    """

    users_numba = List()
    for user in users:
        user_numba = types.int32(user)
        users_numba.append(user_numba)

    node_layer_dict_numba = Dict.empty(
        key_type=types.int32,
        value_type=types.int32[:]
    )   
    for k, v in node_layer_dict.items():
        k = types.int32(k)
        node_layer_dict_numba[k] = np.asarray(v, dtype=np.int32)

    layers_numba = List()
    for layer in layers: 
        layer_numba = Dict.empty(
            key_type=types.int32,
            value_type=types.int32[:]
        )
        for k, v in layer.items():
            k = types.int32(k)
            layer_numba[k] = np.asarray(v, dtype=np.int32)
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
