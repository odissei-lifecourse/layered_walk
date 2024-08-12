
import timeit 
import asyncio
import argparse
import numpy as np
import os 
from math import log2

from src.utils import batched
from dataclasses import dataclass
from typing import Optional
from time import perf_counter
from contextlib import asynccontextmanager


from src.utils import load_data, convert_to_numba

from src.walks import create_walks as create_walks_python
from src.walks import single_walk as single_walk_python

from src.walks_numba import single_walk as single_walk_numba
from src.walks_numba import create_walks as create_walks_numba


# Source: https://esciencecenter-digital-skills.github.io/parallel-python-workbench/extra-asyncio.html
@dataclass
class Elapsed:
    time: Optional[float] = None


@asynccontextmanager
async def timer():
    e = Elapsed()
    t = perf_counter()
    yield e
    e.time = perf_counter() - t



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", 
        dest="dry_run", 
        help="If given, runs a test with a small output data size.",
        action=argparse.BooleanOptionalAction
        )  
    parser.add_argument("--location", help="Snellius or local machine", choices=["snellius", "local"])
    return parser.parse_args()


config_dict = {
    "big": {
        "layers":  ["classmate", "household", "family", "colleague", "neighbor"],
        "walk_len": 50,
        "sample_size": 900_000
    },
    "small": {
        "layers": ["neighbor", "colleague"],
        "walk_len": 5,
        "sample_size": 10_000
    }
}

data_dir = {
    "snellius": "/projects/0/prjs1019/data/graph/processed/",
    "local": "/home/flavio/datasets/synthetic_layered_graph_1mil/"
}


async def main():

    N_RUNS = 10 # number of timing runs

    args = parse_args()
    DRY_RUN = args.dry_run
    DATA_DIR = data_dir[args.location]
    config = config_dict["big"]
    if DRY_RUN:
        config = config_dict["small"]

    SAMPLE_SIZE = config["sample_size"]
    WALK_LEN = config["walk_len"]
    LAYERS = config["layers"]

    print("loading data")
    users, layers, node_layer_dict = load_data(DATA_DIR, LAYERS)
    rng = np.random.default_rng(seed=95359385252)
    users = list(rng.choice(users, size=SAMPLE_SIZE))

    # In order to use numba, we need to store the data in numba-compatible objects
    print("converting to numba")
    users_numba, layers_numba, node_layer_dict_numba = convert_to_numba(users, layers, node_layer_dict)
    

    cpus_avail = os.sched_getaffinity(0)
    print(f"Have the following CPU cores: {cpus_avail}") 
    N_WORKERS = len(cpus_avail) # number of workers to parallelize over
    if DRY_RUN:
        N_WORKERS = N_WORKERS // 2


    # ## walks for a single node 

    # ### Python
    print("timing single run")
    def wrapper():
        return single_walk_python(10, WALK_LEN, node_layer_dict, layers)
    t_single_python = timeit.timeit(wrapper, number=N_RUNS)

    # ### Numba 
    _ = single_walk_numba(10, 5, node_layer_dict_numba, layers_numba) # compile
    def wrapper():
        return single_walk_numba(10, WALK_LEN, node_layer_dict_numba, layers_numba)
    t_single_numba = timeit.timeit(wrapper, number=N_RUNS)

    print(f"single run numba/python: {t_single_numba/t_single_python}")

    # ## Walks for multiple nodes
    print("timing multiple runs")
    def wrapper():
        return create_walks_python(users, WALK_LEN, node_layer_dict, layers)
    #t_mult_python = timeit.timeit(wrapper, number=N_RUNS)

    _ = create_walks_numba(users_numba[:5], 5, node_layer_dict_numba, layers_numba)
    def wrapper():
        return create_walks_numba(users_numba, WALK_LEN, node_layer_dict_numba, layers_numba)
    t_mult_numba = timeit.timeit(wrapper, number=N_RUNS)
    
    print(f"multiple runs, absolute for numba: {t_mult_numba}")
    #print(f"multiple runs, absolute: {t_mult_python} for python, {t_mult_numba} for numba")
    #print(f"multiple runs numba/python: {t_mult_numba/t_mult_python}")


    print("timing parallel runs")
    def walks_wrapper(users):
        return create_walks_numba(users, WALK_LEN, node_layer_dict_numba, layers_numba, 0.8)

    _ = walks_wrapper(users[:10])

    async def create_walks_parallel(users, n_workers):
        result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users, len(users)//n_workers)))
        return result 
    
    workers = [2**i for i in range(int(log2(N_WORKERS))+1)]
    if DRY_RUN:
        workers = [N_WORKERS]
    
    times = {}
    for n_workers in workers:
        data = []
        for _ in range(N_RUNS):
            async with timer() as t:  # not sure this parallelizes. speed is very volatile 
                result = await create_walks_parallel(users_numba, n_workers)
           
            data.append(t.time)

        times[n_workers] = np.mean(data)
    
    print(f"n workers and times: {times}")


    n_walks = sum(len(x) for x in result)
    walk_lengths = [len(x) for x in result[0]]
    avg_lengths = sum(walk_lengths) / len(walk_lengths) # TODO: it seems to fail sometimes! does it have to do with the network type?

    print(f"we created {n_walks} walks with average length of {avg_lengths}")



if __name__ == "__main__":
    asyncio.run(main())


