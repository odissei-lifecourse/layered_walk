
import asyncio
import argparse
import numpy as np
from pathlib import Path
import csv 

from src.utils import (
    batched,
    load_data,
    convert_to_numba,
    get_n_cores
) 
from src.walks_numba import create_walks as create_walks_numba
from src.walks import  create_walks_starting_from_layers
from config import data_dir


LAYERS = ["classmate", "household", "family", "colleague", "neighbor"]
LOCATION_CHOICES = ["snellius", "local", "ossc"]

SAMPLE_SIZE_DRY_RUN = 10_000
LAYERS_DRY_RUN = ["household", "classmate"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", 
        dest="dry_run", 
        help="If given, runs a test with a small output data size.",
        action=argparse.BooleanOptionalAction
        )  
    parser.add_argument("--location", help="Snellius or local machine", choices=LOCATION_CHOICES)
    parser.add_argument("--dest", help="Destination of csv file, relative to data_dir. year will be appended to the end.", type=str)
    parser.add_argument("--n_walks", help="Number of walks per node", type=int, default=5)
    parser.add_argument("--walk_len", help="Length of walks to generate", type=int, default=50)
    parser.add_argument("--year", help="Which year of the network data to use", type=int, default=2010)
    return parser.parse_args()



async def main():

    args = parse_args()
    DRY_RUN = args.dry_run
    LOCATION = args.location
    DATA_DIR = data_dir[LOCATION]
    N_WALKS = args.n_walks
    WALK_LEN = args.walk_len
    YEAR = args.year
    DEST = args.dest
    JUMP_PROB = 0.8

    layers_to_load = LAYERS
    if DRY_RUN:
        layers_to_load = LAYERS_DRY_RUN
    sample_size = -1
    if DRY_RUN:
        sample_size = SAMPLE_SIZE_DRY_RUN

    print("loading data")    
    connected_node_file = "connected_user_set" if LOCATION == "ossc" else None
    users, layer_edge_dict, layer_id_set = load_data(
        DATA_DIR["input"], YEAR, connected_node_file, layers_to_load, sample_size 
    )
    
    for edge_list in layer_edge_dict.values():
        for layer in edge_list.items():
            if len(layer) == 0:
                raise RuntimeError("found an empty python edge list")

    print("converting to numba")
    users_numba, layer_edge_dict_numba = convert_to_numba(users, layer_edge_dict)
    
    for edge_list in layer_edge_dict_numba.values():
        for layer in edge_list.items():
            if len(layer) == 0:
                raise RuntimeError("found an empty numba edge list")

    N_WORKERS = get_n_cores(DRY_RUN)

    def walks_wrapper(users):
        return create_walks_numba(users, WALK_LEN, layer_edge_dict_numba, JUMP_PROB)

    _ = walks_wrapper(users[:10])

    async def create_walks_parallel(users, n_workers):
        result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users, len(users)//n_workers)))
        return result 

    print("Creating walks")
    result = await create_walks_parallel(np.tile(users_numba, N_WALKS), N_WORKERS)
    
    breakpoint()
    additional_walks = create_walks_starting_from_layers(
            layer_id_set=layer_id_set,
            users=users,
            walk_len=WALK_LEN,
            layer_edge_dict=layer_edge_dict,
            p=JUMP_PROB
            ) 
    
    # TODO: append additional_walks to result

    print("Saving")
    filename = DATA_DIR["output"] + DEST + "_" + str(YEAR)
    if DRY_RUN:
        filename += "_dry"

    with Path(filename + ".csv").open("w") as csv_file:
        sample_walk = result[0][0]
        sample_walk_len = len(sample_walk)
        writer = csv.writer(csv_file, delimiter=",")
        header_row = ["SOURCE"] + ["STEP_" + str(i) for i in range(sample_walk_len-1)]
        writer.writerow(header_row)
        for walks in result:
            writer.writerows(walks)



if __name__ == "__main__":
    asyncio.run(main())


