Some experiments to run random walks on a graph with multiple layers with numba


Timing performance (1 random walk for 100k nodes) on fake data:
- pure python (sequential): 104s
- numba (sequential): 4.7s
- numba multithreaded (64 cores): 0.8s

