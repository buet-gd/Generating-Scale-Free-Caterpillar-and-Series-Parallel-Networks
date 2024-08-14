import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random

SEED = 43
DEBUG_MODE = False
random.seed(SEED)


def debug(*args):
    if DEBUG_MODE:
        print("Debug: ", end="")
        print(*args, flush=True)



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

    def has_no_self_loops(graph):
        for edge in graph.edges:
            if edge[0] == edge[1]:
                return False
        return True

    def draw_graph(self):
        G = nx.MultiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)

        fig, ax = plt.subplots(figsize=(20, 10))
        pos = nx.spring_layout(G)  # positions for all nodes
        # Draw nodes and labels
        nx.draw_networkx_nodes(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        # Draw edges with different colors for each edge
        for i, edge in enumerate(G.edges()):
            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=f"C{i}", ax=ax)

        plt.show()

    def get_clustering_coefficient(self):
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.edges)
        return nx.average_clustering(G)

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
    zeta_of_gamma = sum([i ** (-gamma) for i in range(2, 1000000)])
    p_k = k ** (-gamma) / zeta_of_gamma
    return p_k


def generate_power_law_series(num_elements, gamma: float = 2.47):
    degree_values = list(range(2, num_elements))
    probabilities = [calculate_p_k(k, gamma) for k in degree_values]

    integer_list = random.choices(degree_values, probabilities, k=num_elements)
    integer_list = sorted(integer_list, reverse=True)
    if sum(integer_list) % 2 != 0:
        integer_list[0] += 1
    return integer_list


def generate_node_degree_pair_list(num_nodes, gamma: float = 2.47):
    node_degree_pair_list = []
    degree_list = generate_power_law_series(num_nodes, gamma)
    for i in range(num_nodes):
        node_degree_pair_list.append((i, degree_list[i]))
    return node_degree_pair_list


def generate_scale_free_series_parallel_graph(N: int, gamma : float) -> Graph:
    graph = Graph()

    # generate a degree sequence of size N using gamma
    node_degree_pair_list = generate_node_degree_pair_list(N, gamma)

    debug(f"{node_degree_pair_list=}")

    for node, degree in node_degree_pair_list:
        graph.add_node(node)
        graph.degrees[node] = degree

    # split the nodes into two sets U and V

    U = [x for x in graph.nodes if graph.deg(x) > 2]
    V = [x for x in graph.nodes if graph.deg(x) == 2]
    debug(f"{U=}")
    debug(f"{V=}")

    # adjust the largest degree of U so that a graph can be constructed

    open_edge = 0
    open_edge += graph.deg(U[0])

    for i in range(1, len(U)):
        current_edge_degree = graph.deg(U[i])
        debug(f"{open_edge=}  {current_edge_degree=}")
        new_edge_added = min(current_edge_degree - 1, open_edge)
        open_edge -= new_edge_added
        open_edge += current_edge_degree - new_edge_added

    graph.degrees[U[0]] += open_edge

    assert graph.deg(U[0]) > 0

    # main algorithm

    # crate a new stack S
    S = []
    for i in range(graph.deg(U[0])):
        S.append(U[0])

    for i in range(1, len(U)):
        current_node = U[i]
        available_current_degree = graph.deg(current_node)

        while available_current_degree > 1 and len(S) > 0:
            graph.add_edge((current_node, S.pop()))
            available_current_degree -= 1

        while available_current_degree > 0:
            S.append(current_node)
            available_current_degree -= 1

    if S != [] and S[0] == S[1]:
        S.pop()
        S.pop()
        graph.add_edge((U[-1], U[-2]))
        graph.degrees[U[-1]] -= 1
        graph.degrees[U[-2]] += 1

    elif S != []:
        graph.add_edge((S.pop(), S.pop()))

    assert len(S) == 0

    # now add the V nodes

    for v in V:
        random_edge = random.choice(graph.edges)
        graph.add_edge((v, random_edge[0]))
        graph.add_edge((v, random_edge[1]))
        graph.edges.remove(random_edge)

    return graph
