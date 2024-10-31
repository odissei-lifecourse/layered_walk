Some experiments to run random walks on a graph with multiple layers with `numba`


Timing performance (1 random walk for 200k nodes) on fake data:
- pure python (sequential): 209s
- `numba` (sequential): 10.5s
- `numba` multithreaded 
  - 32 cores: 0.82s
  - 64 cores: 0.65s

On real data (NL population network):
- For 15.2 million nodes, creating 1 walk per node of length 15 and saving the result takes about 110s and 190G of memory.


### Sample usage

On Snellius with fake data

```bash
source 2023_snel_modules.sh
source .venv/bin/activate 

export NUMEXPR_MAX_THREADS=32
python create_walks.py \
    --location snellius \
    --year 2010 \
    --n_walks 4 \
    --walk_len 20 \
    --iteration_name first_trial \
    --no-record_edge_types
```

