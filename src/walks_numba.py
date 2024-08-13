import numba 
import numpy as np
from numba.typed import List
from numba.core import types


@numba.njit(nogil=True)
def custom_sample(choice_set: list):
    "custom function to apply np.random.choice"
    if len(choice_set) == 0:
        return -1 
    elif len(choice_set) == 1:
        chosen = choice_set[0]
    else:
        chosen = np.random.choice(choice_set)
    
    return np.int64(chosen)

@numba.njit(nogil=True)
def create_walks(
    nodes: numba.typed.List,
    walk_len: int,
    node_layer_dict: numba.typed.Dict,
    layers: numba.typed.List,
    p: float=0.8
    ):
    result = []
    for node in nodes:
        res = single_walk(
            node, 
            walk_len,
            node_layer_dict,
            layers,
            p
        )
        result.append(res)
    return result



@numba.njit(nogil=True)
def single_walk(start_node: types.int64,
                walk_len: int, 
                node_layer_dict: numba.typed.Dict, 
                layers: numba.typed.List,
                p: float=0.8):
    """Create a single random walk starting at one node.
    
    Args:
        start_node: the node from which to start
        walk_len: the length of the random walk 
        node_layer_dict: dictionary indicating the layer indices in which each node as at least one edge.
        layers: list of numba.typed.Dict. Each layer is an edge list, indicating the connected nodes for each node. 
        p: probability of resampling the layer. 
    
    Returns:
        list: a sequence of node identifiers
    """
    current_node = start_node

    walk = List.empty_list(types.int64)
    walk.append(start_node)


    layer_indices = node_layer_dict[current_node]
    layer_index = custom_sample(layer_indices)
    if layer_index == -1:
        return walk

    for draw in np.random.rand(walk_len):
        layer_indices = node_layer_dict[current_node]

        if draw > p:
            layer_index = custom_sample(layer_indices)
            if layer_index == -1:
                break

        current_layer = layers[layer_index]
        adjacent_nodes = current_layer[current_node]

        walk.append(-layer_index - 1) # the first node is indicated by 0
        next_node = custom_sample(adjacent_nodes)
        if next_node == -1:
            break
        
        walk.append(next_node)
        current_node = next_node

    return walk 
