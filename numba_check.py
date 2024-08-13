
import timeit 
import asyncio
import argparse
import numpy as np
import os 
from math import log2

from src.utils import (
    batched,
    load_data,
    convert_to_numba,
    get_n_cores
)
from src.walks import (
    create_walks as create_walks_python,
    single_walk as single_walk_python
)
from src.walks_numba import (
    create_walks as create_walks_numba,
    single_walk as single_walk_numba
)

from src.async_timing import timer as async_timer
from config import data_dir, config_dict 


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", 
        dest="dry_run", 
        help="If given, runs a test with a small output data size.",
        action=argparse.BooleanOptionalAction
        )  
    parser.add_argument("--location", help="Snellius or local machine", choices=["snellius", "local"])
    parser.add_argument("--year", help="Which year of the network data to use", type=int, default=2010)

    return parser.parse_args()




async def main():

    N_RUNS = 10 # number of timing runs

    args = parse_args()
    DRY_RUN = args.dry_run
    LOCATION = args.location
    DATA_DIR = data_dir[LOCATION]
    YEAR = args.year

    suffix = "_ossc" if LOCATION == "ossc" else ""
    key = f"big{suffix}"

    if DRY_RUN:
        key = f"small{suffix}"
    
    config = config_dict[key]

    SAMPLE_SIZE = config["sample_size"]
    WALK_LEN = config["walk_len"]
    LAYERS = config["layers"]

    N_WORKERS = get_n_cores(DRY_RUN)

    print("loading data")
    connected_node_file = "connected_user_set" if LOCATION == "ossc" else None
    users, layers, node_layer_dict = load_data(
        DATA_DIR, YEAR, connected_node_file, LAYERS, SAMPLE_SIZE
    )

    # In order to use numba, we need to store the data in numba-compatible objects
    print("converting to numba")
    users_numba, layers_numba, node_layer_dict_numba = convert_to_numba(users, layers, node_layer_dict)


    # ## walks for a single node 

    # ### Python
    print("timing single run")
    def wrapper():
        return single_walk_python(10, WALK_LEN, node_layer_dict, layers)
    t_single_python = timeit.timeit(wrapper, number=N_RUNS)

    # ### Numba 
    _ = single_walk_numba(users_numba[0], 5, node_layer_dict_numba, layers_numba) # compile
    def wrapper():
        return single_walk_numba(users_numba[0], WALK_LEN, node_layer_dict_numba, layers_numba)
    t_single_numba = timeit.timeit(wrapper, number=N_RUNS)

    print(f"single run numba/python: {t_single_numba/t_single_python}")

    # ## Walks for multiple nodes
    print("timing multiple runs")
    def wrapper():
        return create_walks_python(users, WALK_LEN, node_layer_dict, layers)
    t_mult_python = timeit.timeit(wrapper, number=N_RUNS)

    _ = create_walks_numba(users_numba[:5], 5, node_layer_dict_numba, layers_numba)
    def wrapper():
        return create_walks_numba(users_numba, WALK_LEN, node_layer_dict_numba, layers_numba)
    t_mult_numba = timeit.timeit(wrapper, number=N_RUNS)
    
    print(f"multiple runs, absolute: {t_mult_python} for python, {t_mult_numba} for numba")
    print(f"multiple runs numba/python: {t_mult_numba/t_mult_python}")


    print("timing parallel runs")
    def walks_wrapper(nodes):
        return create_walks_numba(nodes, WALK_LEN, node_layer_dict_numba, layers_numba, 0.8)

    _ = walks_wrapper(users[:10])

    async def create_walks_parallel(users, n_workers):
        result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users, len(users)//n_workers)))
        return result 
    
    workers = [2**i for i in range(int(log2(N_WORKERS))+1)]
    workers = [workers[0]] + workers[-2:]
    if DRY_RUN:
        workers = [N_WORKERS]
    
    times = {}
    for n_workers in workers:
        data = []
        for _ in range(N_RUNS):
            async with async_timer() as t:  # not sure this parallelizes. speed is very volatile 
                result = await create_walks_parallel(users_numba, n_workers)
           
            data.append(t.time)

        times[n_workers] = np.mean(data)
    
    print(f"n workers and times: {times}")


    n_walks = sum(len(x) for x in result)
    data = []
    for walks in result:
        walk_lengths = [len(x) for x in walks]
        avg_lengths = sum(walk_lengths) / len(walk_lengths) # TODO: it seems to fail sometimes! does it have to do with the network type?
        data.append(avg_lengths)

    avg_lengths = sum(data) / len(data)

    print(f"we created {n_walks} walks with average length of {avg_lengths}")



if __name__ == "__main__":
    asyncio.run(main())


