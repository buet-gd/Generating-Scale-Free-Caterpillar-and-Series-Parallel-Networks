import random
import time
from util import generate_scale_free_series_parallel_graph

SEED = 43
random.seed(SEED)

fhand = open("result.csv", "w")
fhand.write(
    "num_of_nodes,gamma,clustering_coefficient\n"
)


test_samples = [1000, 2000, 3000, 4000, 5000]

print("Starting tests")

for gamma in range(21, 30, 1):
    gamma = gamma/10
    for i in test_samples:
        start_time = time.time()
        print(f"Generating scale-free caterpillar graph with N = {i} with gamma = {gamma}...")
        graph = generate_scale_free_series_parallel_graph(i, gamma)
        out_str = f"{i},{gamma},{graph.get_clustering_coefficient()}\n"
        fhand.write(out_str)

        print(f"Time taken: {(time.time() - start_time)/60} minutes")
        print("\n")

