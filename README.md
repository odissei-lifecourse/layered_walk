Some experiments to run random walks on a graph with multiple layers with `numba`


Timing performance (1 random walk for 200k nodes) on fake data:
- pure python (sequential): 209s
- `numba` (sequential): 10.5s
- `numba` multithreaded 
  - 32 cores: 0.82s
  - 64 cores: 0.65s

On real data (NL population network):
- For 15.2 million nodes, creating 1 walk per node of length 15 and saving the result takes about 110s and 190G of memory.

