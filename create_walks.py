
import asyncio
import argparse
import numpy as np
import logging 

from src.utils import (
    batched,
    load_data,
    convert_to_numba,
    get_n_cores,
    check_layer_edge_dict,
    save_to_parquet
) 
from src.walks_numba import create_walks as create_walks_numba
from src.walks_numba import  create_walks_starting_from_layers
from config import data_dir
from tqdm import tqdm

logger = logging.getLogger(__name__)

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
    parser.add_argument("--debug", help="Debugging. Do additional checks.", 
            default=False, action=argparse.BooleanOptionalAction)
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
    DEBUG = args.debug
    JUMP_PROB = 0.8

    logging_level = logging.DEBUG if DEBUG else logging.INFO
    logging.basicConfig(
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging_level
            )

    layers_to_load = LAYERS
    if DRY_RUN:
        layers_to_load = LAYERS_DRY_RUN
    sample_size = -1
    if DRY_RUN:
        sample_size = SAMPLE_SIZE_DRY_RUN

    logger.info("Loading data")    
    connected_node_file = "connected_user_set" if LOCATION == "ossc" else None
    users, layer_edge_dict, layer_id_set = load_data(
        DATA_DIR["input"], YEAR, connected_node_file, layers_to_load, sample_size 
    )
    layer_id_set = np.array(list(layer_id_set))
    
    if DEBUG:
        check_layer_edge_dict(layer_edge_dict)

    logger.info("Converting to numba")
    users_numba, layer_edge_dict_numba = convert_to_numba(users, layer_edge_dict)
    
    if DEBUG:
        check_layer_edge_dict(layer_edge_dict_numba)

    N_WORKERS = get_n_cores(DRY_RUN)

    def walks_wrapper(users):
        return create_walks_numba(users, WALK_LEN, layer_edge_dict_numba, JUMP_PROB)

    _ = walks_wrapper(users_numba[:10])
    _ = create_walks_starting_from_layers(
            layer_id_set=layer_id_set,
            nodes=users_numba,
            walk_len=WALK_LEN,
            layer_edge_dict=layer_edge_dict_numba,
            p=0.3)

    async def create_walks_parallel(users, n_workers):
        result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users, len(users)//n_workers)))
        return result 
    

    users_input = users_numba
    if LOCATION == "snellius" and not DRY_RUN:
        logger.info("Inflating the work by factor 16")
        users_input = np.tile(users_input, 16) 
    
    for i in tqdm(range(N_WALKS), desc="Creating walks"):
        result = await create_walks_parallel(users_input, N_WORKERS)
        
        logger.debug("Concatenating walks")
        result_array = np.vstack(result)

        logger.info("creating additional walks")
        additional_walks = create_walks_starting_from_layers(
                layer_id_set=layer_id_set,
                nodes=users_numba,
                walk_len=WALK_LEN,
                layer_edge_dict=layer_edge_dict_numba,
                p=JUMP_PROB
                ) 

        logger.info("Concatenating arrays")
        result_array = np.vstack([result_array, additional_walks])

        logger.debug("Saving")
        filename = DATA_DIR["output"] + DEST + "_" + str(YEAR) + "_" + str(i)
        if DRY_RUN:
            filename += "_dry"

        save_to_parquet(result_array, filename)


    logger.info("Done.")



if __name__ == "__main__":
    asyncio.run(main())


