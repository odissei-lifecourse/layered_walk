
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
        return walk # TODO: raise ValueError here? we want all walks to be of the same length?

    for draw in np.random.rand(walk_len):
        layer_indices = list(layer_edge_dict[current_node].keys())
        #layer_indices = node_layer_dict[current_node]

        if draw > p or layer_index not in layer_indices: # because graph is not directed, a node may be reachable on one layer but does not have any outgoing connections on that layer
            layer_index = custom_sample(layer_indices)
            if layer_index == -1:
                break

        adjacent_nodes = layer_edge_dict[current_node][layer_index]
        #current_layer = layers[layer_index]
        #adjacent_nodes = current_layer[current_node]

        walk.append(layer_index)
        next_node = custom_sample(adjacent_nodes)
        if next_node == -1:
            break
        
        walk.append(next_node)
        current_node = next_node

    return walk 


def create_walks(users: list, 
                 walk_len: int,
                 layer_edge_dict: dict,
                 #node_layer_dict: dict,
                 #layers: list,
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
        layer_edge_dict: dict,
        p: float=0.8):
    """"Create one walk for each unique layer identifier."""

    walks = []
    for current_layer in layer_id_set:
        np.random.shuffle(users)
        #walk = [current_layer]
        start_node = None
        while start_node is None:
            for current_user in users:
                if current_layer in layer_edge_dict[current_user]:
                    start_node = current_user
         
        if start_node is None:
            msg = f"Checked all nodes, and none of them had a connection in layer {current_layer}"
            raise RuntimeError(msg)
        
        # invoke the walk function here with walk_len -1; we might also need to crop the last entry in the created walk
        regular_walk = single_walk(
                start_node=start_node, 
                walk_len=walk_len-1,
                layer_edge_dict=layer_edge_dict,
                start_layer=current_layer,
                p=p)
        walk = [current_layer] + regular_walk
         
        required_length = walk_len + (walk_len-1)
        walk = walk[:required_length]
        walks.append(walk)
    
    return walks

    




# def generate_walks(unique_users: list, walk_len: int, p: float = 0.8):
#     "Generate one random walk for each user"
#     num_users = len(unique_users)
#     random_nums = np.random.rand(num_users, walk_len)

#     rows = []
#     for user_idx, user in enumerate(unique_users):   
#         current_node = user
        
#         layer_indices = node_layer_dict[current_node]
#         layer_index = custom_sample(layer_indices)
#         if not layer_index:
#             break

#         current_layer = layers[layer_index]
             
#         walk = [user]
#         while len(walk) < walk_len:
#             layer_indices = node_layer_dict[current_node]
        
#             roll = random_nums[user_idx][len(walk)]
            
#             if roll > p:
#                 layer_index = custom_sample(layer_indices)
#                 if not layer:
#                     break
                
#             adjacent_nodes = current_layer[current_node]

#             # Layer index should encode the layer type in an integer 0-4
#             walk.append(layer_index)
            
#             next_node = custom_sample(adjacent_nodes)
#             if not next_node:
#                 break
            
#             walk.append(next_node)
#             current_node = next_node
#         #assert len(walk) == walk_len, print(len(walk))
#         rows.append(walk)

#     return rows

    
# def create_many_walks(users: list, walk_len: int, num_walks: int):
#     out = []
#     for _ in range(num_walks):
#         walks = generate_walks(unique_users=users, walk_len=walk_len)
#         out.append(walks)
#     return out 
