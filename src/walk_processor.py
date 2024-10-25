"""Tools for generating large numbers of walks and saving them to parquet."""

import asyncio

from src.utils import (
    batched, 
)
from src.walks import create_walks_starting_from_layers
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Callable, AsyncIterator


async def parallel_walks_generator(walk_fct: Callable, users, n_workers: int) -> AsyncIterator[dict[str, int]]:
    """Generate walks in parallel.

    Args:
        `walk_fct`: the function creating the random walks. The function needs to iterate over `users`.
        `users`: the identifiers of the nodes from which walks start.
        `n_workers`: the number of cores for parallel processing.

    Returns:
        A generator with dictionaries of walks. The keys are column names, the values
        are the contents of the rows.
    """
    coroutines = [
        asyncio.to_thread(walk_fct, batch)
        for batch in batched(users, len(users) // n_workers)
    ]

    for future in asyncio.as_completed(coroutines):
        result_list = await future
        for result in result_list:
            yield {"SOURCE": result[0], **{f"STEP_{i}": step for i, step in enumerate(result[1:])}}


async def process_nodes(
        walk_fct: Callable, 
        users, # TODO: add type annotation. numba list? 
        layer_id_set,
        layer_edge_dict,
        walk_len: int, 
        n_workers: int, 
        filename: str,
        jump_prob: float
        ) -> None:
    """Create a single walk for each user and save in a parquet file.
    An additional walk, each one starting at one of the layers, is 
    also added to the result.

    Args:
        `walk_fct`: the function creating the random walks.
        `users`: the identifiers of the nodes from which walks start.
        `layer_id_set`: set of layer identifiers.
        `layer_edge_dict`: dictionary of adjacency dict for each node.
        `n_workers`: the number of cores for parallel processing.
        `filename`: file to save, without suffix.
        `walk_len`: length of walks to generate.
        `jump_prob`: probability of changing layers.
    """
    CHUNK_SIZE = 100_000

    # define schema
    field_col0 = [pa.field("SOURCE", pa.int64())] 
    other_fields = [pa.field(f"STEP_{i}", pa.int64()) for i in range(2*walk_len)]
    fields = field_col0 + other_fields
    schema = pa.schema(fields)

    results = [] 
    chunk_count = 0
    writer = None
    async for result in parallel_walks_generator(walk_fct, users, n_workers):
        results.append(result)

        chunk_count += 1
        if chunk_count >= CHUNK_SIZE:
            writer = await write_chunk(results, schema, filename, writer)
            results.clear()
            chunk_count = 0

    if results:
        writer = await write_chunk(results, schema, filename, writer)

    # add walks starting at node ids
    additional_walks = create_walks_starting_from_layers(
            layer_id_set=layer_id_set,
            users=users,
            walk_len=walk_len,
            n_walks=1,
            layer_edge_dict=layer_edge_dict,
            p=jump_prob)

    add_walks_writeable = []
    for result in additional_walks:
        temp_dict = {"SOURCE": result[0], **{f"STEP_{i}": step for i, step in enumerate(result[1:])}}
        add_walks_writeable.append(temp_dict)

    writer = await write_chunk(add_walks_writeable, schema, filename, writer)

    if writer:
        writer.close()


async def write_chunk(results: list, schema: dict[str, int], filename: str, writer: pq.ParquetWriter | None) -> None:
    table = pa.Table.from_pylist(results)
    if writer is None:
        writer = pq.ParquetWriter(filename + ".parquet", schema)

    writer.write_table(table)
    return writer

