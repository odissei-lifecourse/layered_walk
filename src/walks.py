
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
                node_layer_dict: dict, 
                layers: list,
                p: float=0.8):
    """Create a single random walk starting at one node.
    
    Args:
        start_node: the node from which to start
        walk_len: the length of the random walk 
        node_layer_dict: dictionary indicating the layer indices in which each node as at least one edge.
        layers: list of dicts. Each layer is an edge list, indicating the connected nodes for each node. 
        p: probability of resampling the layer. 
    
    Returns:
        list: a sequence of node identifiers
    """
    current_node = start_node
    walk = [start_node]

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


def create_walks(users: list, 
                 walk_len: int,
                 node_layer_dict: dict,
                 layers: list,
                 p: float=0.8
                 ):
    """Create 1 random walk for each node"""
    walks = []
    for user in users:
        walk = single_walk(start_node=user,
                           walk_len=walk_len,
                           node_layer_dict=node_layer_dict,
                           layers=layers,
                           p=p)
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