
import pickle 
from numba.typed import Dict, List
from numba.core import types
from numba.types import DictType
import numba
import numpy as np
from itertools import islice
from pathlib import Path
from tqdm import tqdm
import warnings
import os 





def load_data(data_dir, 
              year, 
              connected_node_file = None, 
              layer_types: list = ["neighbor", "colleague"],
              sample_size: int = -1
              ):
    """
    Load layered network data from pickle files and process it into a format suitable for further analysis.

    Args:
        data_dir (str): Path to the directory containing the layer data files.
        year (int): Year of the data to use.
        connected_node_file (str, optional): Name of the file containing nodes in the largest connected component.
            Must be stored as "`data_dir`/`connected_node_file`_`year`.pkl".
            If not provided, nodes from the family network will be used as the connected set (use only with fake data).
        layer_types (list, optional): Types of network layers to load. Default is ["neighbor", "colleague"].
            Must be a subset of ["family", "colleague", "classmate", "neighbor", "household"].
        sample_size (int, optional): If positive, returns a random sample of this size from the connected nodes.
            Default is -1 (no sampling).

    Returns:
        tuple: A tuple containing four elements:
            1. list of int: Unique user IDs in the network.
            2. dict: A nested dictionary structure where:
                - The outer key is a user ID.
                - The inner key is a layer ID (maximum user ID + offset + original layer ID).
                - The inner value is a list of connected user IDs for that user in that layer.
            3. set of int: Set of all layer IDs used in the data structure.

    Raises:
        ValueError: If invalid layer types are provided.
        UserWarning: When `connected_node_file` is not provided and family network is used instead.

    Notes:
        - The function loads data from pickle files for each specified layer.
        - It processes the data to create a unified structure across all layers.
        - Layer IDs are assigned by adding an offset to the maximum user ID.
        - If sampling is requested, it's performed on the final set of unique users.
    """


    VALID_LAYERS = ["family", "colleague", "classmate", "neighbor", "household"]
    OFFSET = 5

    if not all([layer in VALID_LAYERS for layer in layer_types]):
        raise ValueError("Invalid layers selected.")

    if connected_node_file:
        with Path(data_dir + connected_node_file + "_" + str(year) + ".pkl").open("rb") as pkl_file:
            unique_users = list(pickle.load(pkl_file))
    else:
        warnings.warn("connected_node_file not provided; using edges from family network. Do this only with fake data.")
        layer_types = list(set(layer_types + ["family"]))


    layers = []
    for ltype in tqdm(layer_types, desc="Loading layers"):
        with Path(data_dir + ltype + "_" + str(year) + "_adjacency_dict.pkl").open("rb") as pkl_file:
            edges = dict(pickle.load(pkl_file))

            if not connected_node_file and ltype == "family":
                unique_users = list(edges.keys())

            layers.append(edges)


    max_user_id = np.max(unique_users)
    layer_edge_dict = {}
    layer_id_set = set() # keep track of all layer ids so that we can create walks starting from there
    for user in tqdm(unique_users, desc="Creating layer_edge_dict"):
        dict_current_user = {}
        for idx, layer in enumerate(layers):
            layer_id = max_user_id + OFFSET + idx
            if user in layer:
                if len(layer[user]) > 0:
                    dict_current_user[layer_id] = layer[user]
                    layer_id_set.add(layer_id)
        layer_edge_dict[user] = dict_current_user

    if sample_size > 0:
        rng = np.random.default_rng(seed=95359385252)
        unique_users = list(rng.choice(unique_users, size=sample_size))

    return unique_users, layer_edge_dict, layer_id_set



def convert_to_numba(users: list, layer_edge_dict: dict[dict[list]]):
    """Convert python data structures to numba-compatible ones.
    
    Args:
        users: list of node identifiers.
        layer_edge_dict: dictionary of layer-specific edge lists for each node.

    Returns:
        the same objects with data types compatible for numba acceleration.
    
    """

    users_numba = List(users)
    users_numba = numba.int64(users_numba)

    user_dict_type = DictType(types.int64, types.int64[:]) 

    layer_edge_dict_numba = Dict.empty(
            key_type=types.int64,
            value_type=user_dict_type
    )

    for user, layer_dict in layer_edge_dict.items():
        user = types.int64(user)
        dict_numba = Dict.empty(
                key_type=types.int64,
                value_type=types.int64[:]
        )
        for layer_id, edgelist in layer_dict.items():
            layer_id = types.int64(layer_id)
            dict_numba[layer_id] = np.asarray(edgelist, dtype=np.int64)

        layer_edge_dict_numba[user] = dict_numba

    return users_numba, layer_edge_dict_numba


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


def check_layer_edge_dict(layer_edge_dict: Dict):
    for edge_list in layer_edge_dict.values():
        for layer in edge_list.values():
            if len(layer) == 0:
                raise RuntimeError("Found empty edge list")



