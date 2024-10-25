
import numpy as np 

def custom_sample(choice_set: list):
    "custom function to apply np.random.choice"
    if len(choice_set) == 0:
        return -1 
    elif len(choice_set) == 1:
        chosen = choice_set[0]
    else:
        chosen = np.random.choice(choice_set)
    
    return np.int64(chosen)


def single_walk(start_node: int,
                walk_len: int, 
                layer_edge_dict: dict,
                start_layer: int | None=None,
                p: float=0.8):
    """Create a single random walk starting at one node.
    
    Args:
        start_node: the node from which to start
        walk_len: the length of the random walk 
        layer_edge_dict: dictionary where keys are node identifiers and values are layer-specific
        edge list, stored in a dict with {layer id: [connected nodes]}.
        p: probability of resampling the layer. 
    
    Returns:
        list: a sequence of node identifiers
    """
    current_node = start_node
    walk = [start_node]


    if start_layer is None:
        layer_indices = list(layer_edge_dict[current_node].keys())
        layer_index = custom_sample(layer_indices)
    else:
        layer_index = start_layer

    if layer_index == -1:
        msg = f"Invalid layer index for node {current_node} with layer indices {layer_indices}" 
        raise ValueError(msg)

    for draw in np.random.rand(walk_len):
        layer_indices = list(layer_edge_dict[current_node].keys())

        if draw > p or layer_index not in layer_indices: # because graph is not directed, a node may be reachable on one layer but does not have any outgoing connections on that layer
            layer_index = custom_sample(layer_indices)
            if layer_index == -1:
                msg = f"Invalid layer index for node {current_node} with layer_indices {layer_indices}"
                raise ValueError(msg)

        adjacent_nodes = layer_edge_dict[current_node][layer_index]

        walk.append(layer_index)
        next_node = custom_sample(adjacent_nodes)
        if next_node == -1:
            current_edge_dict = layer_edge_dict[current_node]
            msg = f"Invalid next_node {next_node} from adjacent_nodes {adjacent_nodes} from {current_node} with current_edge_dict {current_edge_dict}. layer_index is {layer_index}."
            raise ValueError(msg)
        
        walk.append(next_node)
        current_node = next_node

    return walk 


def create_walks(users: list, 
                 walk_len: int,
                 layer_edge_dict: dict,
                 p: float=0.8
                 ):
    """Create 1 random walk for each node"""
    walks = []
    for user in users:
        walk = single_walk(start_node=user,
                           walk_len=walk_len,
                           layer_edge_dict=layer_edge_dict,
                           p=p)
        walks.append(walk)
    return walks


def create_walks_starting_from_layers(
        layer_id_set: set,
        users: list,
        walk_len: int, 
        n_walks: int,
        layer_edge_dict: dict,
        p: float=0.8):
    """"Create one walk for each unique layer identifier.

    Args:
        layer_id_set (set): Set of unique layer identifiers. One walk starting from each of them
        will be created.
        users (list): List of unique node identifiers.
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
        np.random.shuffle(users)
        start_node = None
        
        while start_node is None:
            for current_user in users:
                if current_layer in layer_edge_dict[current_user]:
                    start_node = current_user
         
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
        walks.append(walk)
    
    return walks

    

