import random
import time
from util import generate_scale_free_caterpillar_graph

SEED = 43
random.seed(SEED)

fhand = open("result.csv", "w")
fhand.write(
    "num_of_nodes,average_shortest_path_length,clustering_coefficient,assortativity,avg_degree_centrality,avg_closeness_centrality,avg_betweenness_centrality\n"
)

# take 100 random samples ranging from 100 to 10000
test_samples = random.sample(range(100, 10000), 100)

print("Starting tests")

for i in test_samples:
    start_time = time.time()
    print(f"Generating scale-free caterpillar graph with N = {i}")
    graph = generate_scale_free_caterpillar_graph(i)
    out_str = graph.print_stats()
    fhand.write(out_str)

    print(f"Time taken: {(time.time() - start_time)/60} minutes")
    print("\n")
