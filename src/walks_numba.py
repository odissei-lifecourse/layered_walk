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
    p: float=0.8,
    record_edge_types: bool=True
    ) -> np.ndarray:
    """Create one random walk for each node in `nodes`.
    
    Args:
        nodes: Array of nodes identifiers from which to start a node.
        walk_len: Length of each walk.
        layer_edge_dict: Dictionary of adjacency dicts for each node.
        p: Probability of resampling the layer.
        record_edge_types: If True, the edge IDs of how two nodes are 
        connected is recorded in the walk. This leads to an effective walk
        length of `1 + 2*walk_len` as opposed to `1 + walk_len`. 

    Returns:
        np.ndarray: A 2-dimensional array where each row is a walk starting
        from a given node.

    Notes:
        - The function calls `convert_nested_list_to_array` at the end, which has quadratic 
        time complexity. The advantage is that the conversion is parallelized 
        across workers, and the `for` loop in numba is very fast.
        In practice, this has been the fastest way to collect the results and store
        them in the parquet files.
        - If the input graph has undirected edges, the generated walks will have a 
        higher effective rate of resampling the layer than what is specified in `p`. 
        This is because layers are also resampled when the current node does not 
        have any outgoing edges in the current layer. 
    """
    result = List()
    for node in nodes:
        res = single_walk(
            node, 
            walk_len,
            layer_edge_dict,
            None,
            p,
            record_edge_types
        )
        result.append(res)

    return convert_nested_list_to_array(result)
    

@numba.njit(nogil=True)
def convert_nested_list_to_array(nested_list):
    """Convert a nested numba list to a numpy array.

    Notes:
        This function has quadratic time complexity. It is best run 
        in parallel by multiple workers.
    """
    first_element = nested_list[0]
    output = np.empty((len(nested_list), len(first_element)), dtype=first_element._dtype)
    for i, v in enumerate(nested_list):
        temp_arr = np.empty(len(v), dtype=v._dtype)
        for j, w in enumerate(v):
            temp_arr[j] = w
        output[i] = temp_arr

    return output



@numba.njit(nogil=True)
def single_walk(start_node: types.int64,
                walk_len: int, 
                layer_edge_dict: numba.typed.Dict,
                start_layer: int | None=None,
                p: float=0.8,
                record_edge_types: bool=True) -> numba.typed.List:
    """Create a single random walk starting at one node.
    
    Args:
        start_node: the node from which to start
        walk_len: the length of the random walk 
        layer_edge_dict: dictionary indicating the layer indices in which each node as at least one edge.
        start_layer: identifier of the first layer.
        p: probability of resampling the layer. 
        record_edge_types: If True, the edge type IDs are recorded between
        two connected nodes.
    
    Returns:
        numba.List: a sequence of node identifiers.
    """
   
    current_node = start_node
    walk = List.empty_list(types.int64)
    walk.append(start_node)


    if start_layer is None:
        layer_indices = np.array(list(layer_edge_dict[current_node].keys()))
        layer_index = custom_sample(layer_indices)
        if layer_index == -1:
            msg = f"Invalid layer index for node {current_node} with layer indices {layer_indices}" 
            raise ValueError(msg)
    else:
        layer_index = start_layer

    
  
    for draw in np.random.rand(walk_len):
        layer_indices = np.array(list(layer_edge_dict[current_node].keys()))

        if draw > p or layer_index not in layer_indices:
            layer_index = custom_sample(layer_indices)
            if layer_index == -1:
                msg = f"invalid layer index for node {current_node} with layer_indices {layer_indices}"
                raise RuntimeError(msg)

        adjacent_nodes = layer_edge_dict[current_node][layer_index]

        if record_edge_types: 
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
        nodes: np.ndarray,
        walk_len: int, 
        layer_edge_dict: numba.typed.Dict,
        p: float=0.8) -> np.ndarray:
    """"Create one walk for each unique layer identifier.

    Args:
        layer_id_set: Array of unique layer identifiers. One walk starting from each of them
        will be created.
        nodes: List of unique node identifiers.
        walk_len: length of the walk to generate.
        n_walks: Number of walks to generate for each layer.
        layer_edge_dict: Numba dictionary where keys are node identifiers and values are dictionaries 
        of non-empty edge lists for each layer.
        p: probability of resampling the layer at each node.

    Returns:
        np.ndarray: A 2-dimensional array where each row is a walk starting from 
        one of the layer identifiers.

    Notes:
        The function calls `single_walk` from a random draw of the nodes that are connected
        on each layer type. The layer identifier is then inserted at the start of 
        the walk.
    """

    walks = List() 
    for current_layer in layer_id_set:
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
        walk = single_walk(
                start_node=start_node, 
                walk_len=walk_len,
                layer_edge_dict=layer_edge_dict,
                start_layer=current_layer,
                p=p)

        walk.insert(0, current_layer)
        
        expected_length = 1 + 2*walk_len
        walk = walk[:expected_length]
        walks.append(walk)
    
    return convert_nested_list_to_array(walks)





