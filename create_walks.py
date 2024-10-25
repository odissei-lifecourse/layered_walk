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
    save_to_file,
)
from src.walks_numba import create_walks as create_walks_numba
from src.walks import create_walks_starting_from_layers
from config import data_dir
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict


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
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--location", help="Snellius or local machine", choices=LOCATION_CHOICES
    )
    parser.add_argument(
        "--dest",
        help="Destination of csv file, relative to data_dir. year will be appended to the end.",
        type=str,
    )
    parser.add_argument(
        "--n_walks", help="Number of walks per node", type=int, default=5
    )
    parser.add_argument(
        "--walk_len", help="Length of walks to generate", type=int, default=50
    )
    parser.add_argument(
        "--year", help="Which year of the network data to use", type=int, default=2010
    )
    parser.add_argument(
        "--debug",
        help="Debugging. Do additional checks.",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
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
        level=logging_level,
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

    if DEBUG:
        check_layer_edge_dict(layer_edge_dict)

    logger.info("Converting to numba")
    users_numba, layer_edge_dict_numba = convert_to_numba(users, layer_edge_dict)

    if DEBUG:
        check_layer_edge_dict(layer_edge_dict_numba)

    N_WORKERS = get_n_cores(DRY_RUN)

    def walks_wrapper(users):
        return create_walks_numba(users, WALK_LEN, layer_edge_dict_numba, JUMP_PROB)

    _ = walks_wrapper(users[:10])

    ### --START this should go into its own file
    async def walks_generator(users, n_workers):
        coroutines = [
            asyncio.to_thread(walks_wrapper, batch)
            for batch in batched(users, len(users) // n_workers)
        ]

        for future in asyncio.as_completed(coroutines):
            result_list = await future
            for result in result_list:
                yield {"SOURCE": result[0], **{f"STEP_{i}": step for i, step in enumerate(result[1:])}}

        # result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users, len(users)//n_workers)))
        # return result

    async def process_nodes(users, n_workers, filename):
        # define schema
        field_col0 = [pa.field("SOURCE", pa.int64())] 
        other_fields = [pa.field(f"STEP_{i}", pa.int64()) for i in range(2*WALK_LEN)]
        fields = field_col0 + other_fields
        schema = pa.schema(fields)

        results = [] 
        chunk_count = 0
        writer = None
        async for result in walks_generator(users, n_workers):
            results.append(result)

            chunk_count += 1
            if chunk_count >= 100_000:
                writer = await write_chunk(results, writer, filename, schema)
                results.clear()
                chunk_count = 0

        if results:
            writer = await write_chunk(results, writer, filename, schema)

        # add walks starting at node ids
        additional_walks = create_walks_starting_from_layers(
                layer_id_set = layer_id_set,
                users=users,
                walk_len=WALK_LEN,
                n_walks=1,
                layer_edge_dict=layer_edge_dict,
                p=JUMP_PROB)

        add_walks_writeable = []
        for result in additional_walks:
            temp_dict = {"SOURCE": result[0], **{f"STEP_{i}": step for i, step in enumerate(result[1:])}}
            add_walks_writeable.append(temp_dict)

        writer = await write_chunk(add_walks_writeable, writer, filename, schema)

        if writer:
            writer.close()


    async def write_chunk(results, writer, filename, schema):
        table = pa.Table.from_pylist(results)
        if writer is None:
            writer = pq.ParquetWriter(filename + ".parquet", schema)

        writer.write_table(table)
        return writer

    ### --END-- this should go into its own file

    logger.info("Creating walks")
    for i in tqdm(range(N_WALKS)):
        filename = DATA_DIR["output"] + DEST + "_" + str(YEAR) + "_" + str(i)
        
        if DRY_RUN:
            filename += "_dry"
        
        #required_length = len(result[0][0])
        #additional_walks = [x[:required_length] for x in additional_walks]
        await process_nodes(users_numba, N_WORKERS, filename)

        #logger.info("Saving")


    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
