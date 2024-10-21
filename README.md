Some experiments to run random walks on a graph with multiple layers with numba


Timing performance (1 random walk for 100k nodes) on fake data:
- pure python (sequential): ~~104s~~ 14.4s
- numba (sequential): ~~4.7s~~ 0.9s
- numba multithreaded (64 cores): ~~0.8s~~ 0.21s

On real data (NL population network):
- For 15.2 Mio nodes, creating 5 walks per node of length 10 takes 1.5 hours and 160GB of memory. 

