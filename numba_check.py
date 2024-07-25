
import timeit 
import asyncio

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


layers_to_use = ["neighbor", "colleague"]
layers_to_use = ["education", "household"]
walk_len = 5
sample_size = 200_000
N_RUNS = 2 # number of timing runs
N_WORKERS = 8 # number of workers to parallelize over


print("loading data")
users, layers, node_layer_dict = load_data(layers_to_use)
# In order to use numba, we need to store the data in numba-compatible objects
print("converting to numba")
users_numba, layers_numba, node_layer_dict_numba = convert_to_numba(users, layers, node_layer_dict)

users, layers, node_layer_dict = load_data(layers_to_use)


async def main():

    # walks for a single node 
    print("timing single run")
    myfunc = "single_walk_python(10, walk_len, node_layer_dict, layers)"
    t_single_python = timeit.timeit(myfunc, globals=globals(), number=N_RUNS)

    # compile 
    _ = single_walk_numba(10, walk_len, node_layer_dict_numba, layers_numba)
    myfunc = "single_walk_numba(10, 5, node_layer_dict_numba, layers_numba)"
    t_single_numba = timeit.timeit(myfunc, globals=globals(), number=N_RUNS)
    print(f"single run numba/python: {t_single_numba/t_single_python}")

    # walks for multiple nodes
    print("timing multiple runs")
    myfunc = "create_walks_python(users[:sample_size], walk_len, node_layer_dict, layers)"
    t_mult_python = timeit.timeit(myfunc, globals=globals(), number=N_RUNS)
    # _ = create_walks_python(users[:sample_size], walk_len, node_layer_dict, layers)
    # compile
    _ = create_walks_numba(users_numba[:sample_size], walk_len, node_layer_dict_numba, layers_numba)
    myfunc = "create_walks_numba(users_numba[:sample_size], walk_len, node_layer_dict_numba, layers_numba)"
    t_mult_numba = timeit.timeit(myfunc, globals=globals(), number=N_RUNS)

    print(f"multiple runs, absolute: {t_mult_python} for python, {t_mult_numba} for numba")
    print(f"multiple runs numba/python: {t_mult_numba/t_mult_python}")

    # _ = create_walks_numba(users_numba[:sample_size], walk_len, node_layer_dict_numba, layers_numba)


    print("timing parallel runs")
    def walks_wrapper(users):
        return create_walks_numba(users, 5, node_layer_dict_numba, layers_numba, 0.8)

    _ = walks_wrapper(users[:10])

    async def create_walks_parallel(users, n, n_workers):
        result = await asyncio.gather(*(asyncio.to_thread(walks_wrapper, batch) for batch in batched(users[:n], n//n_workers)))
        return result 
    
    async with timer() as t:  # not sure this parallelizes. speed is very volatile 
        result = await create_walks_parallel(users, sample_size, N_WORKERS)

    print(f"that took {t.time} seconds")


    n_walks = sum(len(x) for x in result)
    # NOTE: this is not correct b/c there is no padding. (I think) The walk may just stop when there are no more nodes to go to. 
    final_length = len(result[0][0]) # TODO: it seems to fail sometimes! does it have to do with the network type?
    # try:
    #     assert final_length >= walk_len, "walk generation failed" # we also store the identifiers of the layers, which adds elements to the walk
    # except:
    #     breakpoint()

    print(f"we created {n_walks} walks of length {final_length}")



if __name__ == "__main__":
    asyncio.run(main())


