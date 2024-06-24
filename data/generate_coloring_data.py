import argparse
import os
import pickle
from turtle import color
import numpy as np 
import pprint as pp
import time
import tqdm
import networkx as nx 
from multiprocessing import Pool
import warnings
from pysat.solvers import Solver
warnings.filterwarnings("ignore")

# I bet this could be implemented much faster
def get_sat_clauses(g : nx.Graph, num_colors):
    sat_clauses = []
    for v in range(g.number_of_nodes()):
        clause = [v * num_colors + i + 1 for i in range(num_colors)]
        sat_clauses.append(clause)
    for v, u in g.edges():
        for i in range(num_colors):
            sat_clauses.append([-(v * num_colors + i + 1), -(u * num_colors + i + 1)])
    for v in range(g.number_of_nodes()):
        for i in range(num_colors):
            for j in range(i + 1, num_colors):
                sat_clauses.append([-(v * num_colors  + i + 1), -(v * num_colors + j + 1)])
    return sat_clauses

# Could be improved using numpy
def get_coloring_from_sat(n, num_colors, sat_result):
    coloring = [-1] * n
    for value in sat_result:
        v = (value - 1) // num_colors
        color = value - v * num_colors - 1
        if value < 0:
            continue
        if coloring[v] != -1:
            continue
        coloring[v] = color
    return coloring

def do_checks(g, coloring, num_colors):
    if np.max(coloring) > num_colors or np.min(coloring) < 0:
        raise Exception("colorings must be with range [0, {num_colors}]")
    for (v, u) in g.edges():
        if coloring[v] == coloring[u]:
            raise Exception("Nodes {v} and {u} have the same color c")

def solve_coloring(g : nx.Graph, solver, strategy, num_colors):
    if solver == "greedy":
        if opts.strategy is None: 
            coloring =  nx.coloring.greedy_color(g)
        else:
            coloring =  nx.coloring.greedy_color(g, strategy)
    elif solver == "sat":
        with Solver(name=strategy) as s:
            clauses = get_sat_clauses(g, num_colors)
            for clause in clauses:
                s.add_clause(clause, num_colors)
            result = s.solve()
            if result == True:
                coloring = get_coloring_from_sat(g.number_of_nodes(), num_colors, s.get_model())
            else:
                coloring = [num_colors + 1] * g.number_of_nodes() 
                # this indicates that it's impossible to color with up to num_colors colors
        return coloring
    elif solver == "sat_optim":
        raise NotImplementedError("Will do later")
    else:
        raise ValueError(f"Unknown solver: {solver}")
    n = g.number_of_nodes()
    return [coloring[i] for i in range(n)]

if __name__ == "__main__":
    # Parsing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_colors", type=int, default=25)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--solver", type=str, default="greedy")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--do_checks", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--solver_time_limit", type=float, default=None)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

    # Setting the random seed 
    np.random.seed(opts.seed)

    if opts.folder is None:
        opts.folder = f"coloring{opts.min_nodes}-{opts.max_nodes}_" + opts.solver + '_data/'
    assert(opts.folder[-1] == '/')
    os.makedirs(opts.folder)

    # Pretty print the run args
    pp.pprint(vars(opts)) 
    
    sample_number = 0
    start_time = time.time()
    for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
        num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1) 
        assert opts.min_nodes <= num_nodes <= opts.max_nodes

        batch_graphs = []
        colorings = []
        # TODO: fix below
        # with Pool(opts.batch_size) as p: # Parallel processing 
        #     colorings = p.map(solve_coloring, batch_graphs)
        while len(batch_graphs) < opts.batch_size:
            g = nx.erdos_renyi_graph(num_nodes, opts.density)
            if opts.solver != 'planted':
                coloring = solve_coloring(g, opts.solver, opts.strategy, opts.num_colors)
            else:
                coloring = list(np.random.randint(0, opts.num_colors, size = num_nodes))
                new_edges = [(u, v) for (u, v) in g.edges() if coloring[v] != coloring[u]]
                g_new = nx.Graph()
                g_new.add_edges_from(new_edges)
                g = g_new
            if max(coloring) < opts.num_colors:
                batch_graphs.append(g)

                if opts.do_checks == True:
                    do_checks(g, coloring, opts.num_colors) 
                colorings.append(coloring)

        for idx, coloring in enumerate(colorings):
            g_final = nx.Graph() 
            for (v, c) in enumerate(coloring):
                g_final.add_node(v, label = c)
            g_final.add_edges_from(batch_graphs[idx].edges)

            with open(opts.folder + str(sample_number) + '.pickle', 'wb') as f:
                pickle.dump(g_final, f)
            sample_number += 1
            # f.write(str(num_nodes) + " edges ")
            # f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_graphs[idx].edges))
            # f.write(str(" ") + str('output') + str(" "))
            # f.write(str(" ").join(str(c) for c in coloring))
            # f.write("\n")

    end_time = time.time() - start_time

    assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")
