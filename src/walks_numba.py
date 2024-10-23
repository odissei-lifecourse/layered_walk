import numba 
import numpy as np
from numba.typed import List
from numba.core import types


@numba.njit(nogil=True)
def custom_sample(choice_set: List):
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
    nodes: numba.int64[:],
    walk_len: int,
    layer_edge_dict: numba.typed.Dict,
    p: float=0.8
    ):
    result = List()
    for node in nodes:
        res = single_walk(
            node, 
            walk_len,
            layer_edge_dict,
            p
        )
        result.append(res)
    return result



@numba.njit(nogil=True)
def single_walk(start_node: types.int64,
                walk_len: int, 
                layer_edge_dict: numba.typed.Dict,
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

    
    layer_indices = np.array(list(layer_edge_dict[current_node].keys()))
    layer_index = custom_sample(layer_indices)
    if layer_index == -1:
        msg = f"invalid layer index for node {current_node} with layer_indices {layer_indices}"
        raise RuntimeError(msg)

    for draw in np.random.rand(walk_len):
        layer_indices = np.array(list(layer_edge_dict[current_node].keys()))

        if draw > p or layer_index not in layer_indices:
            layer_index = custom_sample(layer_indices)
            if layer_index == -1:
                msg = f"invalid layer index for node {current_node} with layer_indices {layer_indices}"
                raise RuntimeError(msg)

        adjacent_nodes = layer_edge_dict[current_node][layer_index]

        walk.append(layer_index) # the first node is indicated by 0
        next_node = custom_sample(adjacent_nodes)
        if next_node == -1:
            msg = f"Invalid next_node from adjacent nodes {adjacent_nodes} of current node {current_node} in layer_index {layer_index}"
            raise RuntimeError(msg) 
        
        walk.append(next_node)
        current_node = next_node

    return walk 
