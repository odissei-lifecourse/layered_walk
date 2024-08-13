
import asyncio
import argparse
import numpy as np
import os 
from pathlib import Path
import csv 

from src.utils import batched
from src.utils import load_data, convert_to_numba
from src.walks_numba import create_walks as create_walks_numba


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



data_dir = {
    "snellius": "/projects/0/prjs1019/data/graph/processed/",
    "local": "/home/flavio/datasets/synthetic_layered_graph_1mil/",
    "ossc": "/gpfs/ostor/ossc9424/homedir/Dakota_network/intermediates/"
}


async def main():

    args = parse_args()
    DRY_RUN = args.dry_run
    LOCATION = args.location
    DATA_DIR = data_dir[LOCATION]
    N_WALKS = args.n_walks
    WALK_LEN = args.walk_len
    YEAR = args.year
    DEST = args.dest

    LAYERS = LAYERS if not DRY_RUN else LAYERS_DRY_RUN

    print("loading data")
    users, layers, node_layer_dict = load_data(DATA_DIR, YEAR, "connected_user_set", LAYERS)
    if DRY_RUN:
        rng = np.random.default_rng(seed=95359385252)
        users = list(rng.choice(users, size=SAMPLE_SIZE_DRY_RUN))    
    
    print("converting to numba")
    users_numba, layers_numba, node_layer_dict_numba = convert_to_numba(users, layers, node_layer_dict)
    

    cpus_avail = os.sched_getaffinity(0)
    print(f"Have the following CPU cores: {cpus_avail}") 
    N_WORKERS = len(cpus_avail) # number of workers to parallelize over
    if LOCATION == "local":
        N_WORKERS = N_WORKERS // 2

    def walks_wrapper(users):
        return create_walks_numba(users, WALK_LEN, node_layer_dict_numba, layers_numba, 0.8)

    _ = walks_wrapper(users[:10])

    async def create_walks_parallel(users, n_workers):
        result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users, len(users)//n_workers)))
        return result 
    
    result = await create_walks_parallel(N_WALKS * users_numba, N_WORKERS) # TODO: cannot do int * numba.typed.List

    filename = DATA_DIR + DEST + "_" + str(YEAR)
    if DRY_RUN:
        filename += "_dry"

    with Path(filename + ".csv").open("w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        header_row = ["SOURCE"] + ["STEP_" + str(i) for i in range(WALK_LEN-1)]
        writer.writerow(header_row)
        for walks in result:
            writer.writerows(walks)



if __name__ == "__main__":
    asyncio.run(main())


