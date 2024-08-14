import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random

SEED = 43
DEBUG_MODE = False


def debug(*args):
    if DEBUG_MODE:
        print("Debug: ", end="")
        print(*args, flush=True)


try:
    with open("p_k_dict.pickle", "rb") as f:
        p_k_cache = pickle.load(f)
except:
    p_k_cache = None


class Graph:
    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.degrees = {}

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def deg(self, u) -> int:
        return self.degrees[u]

    def draw_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=SEED)
        nx.draw(
            G,
            pos=pos,
            with_labels=True,
            font_size=10,
            node_size=500,
            node_color="skyblue",
            edge_color="gray",
            linewidths=0.5,
            font_color="black",
            ax=ax,
        )

        plt.tight_layout()
        plt.savefig("generated_caterpillar.eps", format="eps", dpi=300)
        plt.show()

    def get_average_shortest_path_length(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        return nx.average_shortest_path_length(G)

    def print_stats(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        num_of_nodes = G.number_of_nodes()
        average_shortest_path_length = nx.average_shortest_path_length(G)
        clustering_coefficient = nx.average_clustering(G)
        assortativity = nx.degree_assortativity_coefficient(G)

        # Centrality measures
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        # eigenvector_centrality = nx.eigenvector_centrality(G)

        avg_degree_centrality = sum(degree_centrality.values()) / num_of_nodes
        avg_closeness_centrality = sum(closeness_centrality.values()) / num_of_nodes
        avg_betweenness_centrality = sum(betweenness_centrality.values()) / num_of_nodes
        # avg_eigenvector_centrality = sum(eigenvector_centrality.values()) / num_of_nodes

        print(
            "num_of_nodes,average_shortest_path_length,clustering_coefficient,assortativity,avg_degree_centrality,avg_closeness_centrality,avg_betweenness_centrality"
        )
        print(
            num_of_nodes,
            average_shortest_path_length,
            clustering_coefficient,
            assortativity,
            avg_degree_centrality,
            avg_closeness_centrality,
            avg_betweenness_centrality,
            sep=",",
        )

        return f"{num_of_nodes},{average_shortest_path_length},{clustering_coefficient},{assortativity},{avg_degree_centrality},{avg_closeness_centrality},{avg_betweenness_centrality}\n"


def calculate_p_k(k: int, gamma: float = 2.47):
    if p_k_cache is not None and k in p_k_cache:
        return p_k_cache[k]
    zeta_of_gamma = sum([i ** (-gamma) for i in range(1, 1000000)])
    p_k = k ** (-gamma) / zeta_of_gamma
    return p_k


def generate_power_law_series(num_elements, gamma: float = 2.47):
    debug(
        f"Generating power law series with {num_elements} elements and gamma = {gamma}"
    )
    degree_values = list(range(1, num_elements))
    probabilities = [calculate_p_k(k, gamma) for k in degree_values]

    integer_list = random.choices(degree_values, probabilities, k=num_elements)
    integer_list = sorted(integer_list, reverse=True)
    if sum(integer_list) % 2 != 0:
        integer_list[0] += 1

    for item in integer_list:
        assert item > 0
    return integer_list


def generate_node_degree_pair_list(num_nodes, gamma: float = 2.47):
    debug(
        f"Generating node degree pair list with {num_nodes} nodes and gamma = {gamma}"
    )
    node_degree_pair_list = []
    degree_list = generate_power_law_series(num_nodes, gamma)
    for i in range(num_nodes):
        node_degree_pair_list.append((i, degree_list[i]))
    return node_degree_pair_list


def fisher_yates_shuffle(arr, randomness_coef: int):
    """
    Shuffle the given array in-place using the Fisher-Yates shuffle algorithm.
    
    Parameters:
    arr (list): The array to be shuffled.
    """
    num_of_itteration = int(len(arr) * randomness_coef)
    for i in range(len(arr) - 1, num_of_itteration, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]



def generate_scale_free_caterpillar_graph(N: int, randomness_coeff: int = 0) -> Graph:
    assert N > 0
    assert randomness_coeff >= 0 and randomness_coeff <= 1
    graph = Graph()

    # generate a degree sequence of size N using gamma
    node_degree_pair_list = generate_node_degree_pair_list(N, gamma=2.47)
    for node, degree in node_degree_pair_list:
        graph.add_node(node)
        graph.degrees[node] = degree

    # split the nodes into two sets U and V

    U = [x for x in graph.nodes if graph.deg(x) > 1]
    V = [x for x in graph.nodes if graph.deg(x) == 1]
    debug(f"{U=}")
    debug(f"{V=}")

    # readjust the degree of the largest node to match the existential condition.
    delta = sum([graph.deg(node) for node in U]) - (2 * len(U)) + 2 - len(V)

    # delta is negative, increase the degree of the largest node
    if delta < 0:
        graph.degrees[U[0]] -= delta

    # delta is positive, decrease the degree of the largest nodes

    while delta > 0:
        for u in U:
            if graph.deg(u) > 2:
                graph.degrees[u] -= 1
                delta -= 1
                if delta == 0:
                    break

    for u in U:
        assert graph.deg(u) > 1

    assert sum([graph.deg(node) for node in U]) - 2 * len(U) + 2 - len(V) == 0

    # random.shuffle(U)
    # debug("U after random.shuffle(U):", U)

    # fisher_yates_shuffle(U, 0.1)

    debug(f"{U=}")

    # create the central path
    for i in range(1, len(U)):
        graph.add_edge((U[i], U[i - 1]))

    # create the leaves
    for u in U:
        for _ in range(graph.deg(u) - 2):
            v = V.pop()
            graph.add_edge((u, v))

    v = V.pop()
    graph.add_edge((U[0], v))

    v = V.pop()
    graph.add_edge((U[-1], v))

    debug(f"{U=}")
    debug(f"{V=}")

    return graph
