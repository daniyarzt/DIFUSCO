""" Utilities for Graph Coloring Problem (GCP) """

import networkx as nx
import typing
import numpy as np

class RandomQueue():
    def __init__(self, items : typing.Iterable):
        self.items = items

    def pop(self):
        if len(self.items) == 0:
            raise IndexError()

        id = np.random.randint(0, len(self.items))
        self.items[-1], self.items[id] = self.items[id], self.items[-1]
        return self.items.pop()

    def push(self, x):
        self.items.append(x)

    def empty(self): 
        return len(self.items) == 0

# TODO: Figure out weather to pass seed to this
def sample_greedy_order_from_coloring(g : nx.Graph, n_colors : int, coloring : list[int]) -> list[int]:
    """Given a graph, returns a permutation of vertices such that the greedy algorithm
    achieves the same coloring (up to the relabeling of colors)."""

    n_vertices = g.number_of_nodes()
    assert(n_vertices == len(coloring))
    is_adj_col = [[False for col in range(n_colors)] for v in range(n_vertices)]

    # Relabeling colors randomly
    col_perm = np.random.permutation(n_colors)
    coloring = list(map(lambda col : col_perm[col], coloring))
    
    # Grouping by color
    group = [[] for i in range(n_colors)]
    for v in range(n_vertices):
        group[coloring[v]].append(v)
    assert(len(group[i]) != 0 for i in range(n_colors))

    # Checks if vertex u can be placed next in the greedy order
    def check_order(u : int) -> bool:
        for col in range(coloring[u]):
            if not is_adj_col[col]:
                return False
        return True    

    # Sampling a greedy order
    q = RandomQueue(group[0])
    used = [v in group[0] for v in range(n_vertices)]
    greedy_order = []
    while not q.empty():
        v = q.pop()
        greedy_order.append(v)
        for u in g.neighbors(v):
            is_adj_col[u][coloring[v]] = True
            if not used[u] and check_order(u):
                q.push(u)
                used[u] = True
    
    return greedy_order
        
def greedy_coloring(g : nx.Graph, order : typing.Iterable) -> list[int]:
    """Given a graph and the greedy order returns the coloring."""
    n_vertices = g.number_of_nodes()
    coloring = [None for i in range(n_vertices)]

    for v in order:
        col_set = set()
        for u in g.neighbors(v):
            if not coloring[u] is None:
                col_set.add(coloring[u])
        coloring[v] = 0
        while coloring[v] in col_set:
            coloring[v] += 1
    
    return coloring

def compare_colorings(g : nx.Graph, coloring1 : list[int], coloring2 : list[int]) -> bool: 
    """Given a graph and two vertex colorings, checks if colorings are isomorphic."""
    n_colors = max(coloring1) + 1
    assert(n_colors == max(coloring2) + 1)
    n_vertices = g.number_of_nodes()
    first_with_col1 = [None for col in range(n_colors)]
    first_with_col2 = [None for col in range(n_colors)]

    for v in range(n_vertices):
        if first_with_col1[coloring1[v]] is None:
            first_with_col1[coloring1[v]] = v
        if first_with_col2[coloring2[v]] is None:
            first_with_col2[coloring2[v]] = v
        if first_with_col1[coloring1[v]] != first_with_col2[coloring2[v]]:
            return False
    return True

def test_sample_order():
    g = nx.Graph()
    for v in range(5):
        g.add_node(v)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(1, 4)
    g.add_edge(1, 3)
    g.add_edge(2, 3)
    g.add_edge(4, 3)
    coloring1 = [0, 1, 2, 0, 2]
    order = sample_greedy_order_from_coloring(g, 3, coloring1)
    coloring2 = greedy_coloring(g, order)
    print(compare_colorings(g, coloring1, coloring2))

if __name__ == '__main__':
    test_sample_order()