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
            None,
            p
        )
        result.append(res)
    
    #return result 
    
    # try converting to numpy
    # pro: numba for loops are fast, plus it's done by the workers. plus,
    # writing becomes very fast too.
    # con: more memory (?); overhead operation but it's faster
        # the memory issue should be checked in more detail: compare
        # memory usage with this process and without (only calling the workers, not writing)
    # than concatenating the result in the main process
    # see github discussion. 
    # TODO: put into a separate function
    A = result[0]
    a = np.empty((len(result), len(A)), dtype=A._dtype)
    for i, v in enumerate(result):
        temp_arr = np.empty(len(v), dtype=v._dtype)
        for j, w in enumerate(v):
            temp_arr[j] = w
        a[i] = temp_arr
    return a




@numba.njit(nogil=True)
def single_walk(start_node: types.int64,
                walk_len: int, 
                layer_edge_dict: numba.typed.Dict,
                start_layer: int | None=None,
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


    if start_layer is None:
        layer_indices = np.array(list(layer_edge_dict[current_node].keys()))
        layer_index = custom_sample(layer_indices)
    else:
        layer_index = start_layer

    if layer_index == -1:
        msg = f"Invalid layer index for node {current_node} with layer indices {layer_indices}" 
        raise ValueError(msg)
    
  
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


@numba.njit(nogil=True)
def create_walks_starting_from_layers(
        layer_id_set: np.ndarray,
        nodes: numba.int64[:],
        walk_len: int, 
        n_walks: int,
        layer_edge_dict: numba.typed.Dict,
        p: float=0.8):
    """"Create one walk for each unique layer identifier.

    Args:
        layer_id_set (np.ndarray): Array of unique layer identifiers. One walk starting from each of them
        will be created.
        nodes (list): List of unique node identifiers.
        walk_len (int): length of the walk to generate.
        n_walks (int): Number of walks to generate for each layer.
        layer_edge_dict: Dictionary where keys are node identifiers and values are dictionaries 
        of non-empty edge lists for each layer.
        p: probability of changing layer.

    Returns:
        list: a list of walks, one starting from each of the layer identifiers.
    """

    walks = []
    for current_layer in np.tile(list(layer_id_set), n_walks):
        np.random.shuffle(nodes)
        start_node = None
        
        while start_node is None:
            for current_node in nodes:
                if current_layer in layer_edge_dict[current_node]:
                    start_node = current_node 
         
        if start_node is None:
            msg = f"Checked all nodes, and none of them had a connection in layer {current_layer}"
            raise RuntimeError(msg)
        
        # invoke the walk function here with walk_len; crop at the end
        regular_walk = single_walk(
                start_node=start_node, 
                walk_len=walk_len,
                layer_edge_dict=layer_edge_dict,
                start_layer=current_layer,
                p=p)
        walk = [current_layer] + regular_walk
        expected_length = 1 + 2*walk_len
        walk = walk[:expected_length]
        walks.append(walk)
    
    return walks





