import argparse
import numpy as np 
import pprint as pp
import time
import tqdm
import networkx as nx 
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Parsing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_colors", type=int, default=25)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--num_samples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--solver", type=str, default="greedy")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

    # Setting the random seed 
    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"coloring{opts.min_nodes}-{opts.max_nodes}_" + opts.solver + ".txt"
    
    # Pretty print the run args
    pp.pprint(vars(opts)) 
    
    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1) 
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            def solve_coloring(g : nx.Graph):
                if opts.solver == "greedy":
                    if opts.strategy is None: 
                        coloring =  nx.coloring.greedy_color(g)
                    else:
                        coloring =  nx.coloring.greedy_color(g, opts.strategy)
                else:
                    raise ValueError(f"Unknown solver: {opts.solver}")
                n = g.number_of_nodes()
                return [coloring[i] for i in range(n)]

            batch_graphs = []
            colorings = []
            # TODO: fix below
            # with Pool(opts.batch_size) as p: # Parallel processing 
            #     colorings = p.map(solve_coloring, batch_graphs)
            while len(batch_graphs) < opts.batch_size:
                g = nx.erdos_renyi_graph(num_nodes, opts.density)
                coloring = solve_coloring(g)
                if max(coloring) < opts.num_colors:
                    batch_graphs.append(g)
                    colorings.append(coloring)

            for idx, coloring in enumerate(colorings):
                f.write(str(num_nodes) + " edges ")
                f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_graphs[idx].edges))
                f.write(str(" ") + str('output') + str(" "))
                f.write(str(" ").join(str(c) for c in coloring))
                f.write("\n")

    end_time = time.time() - start_time

    assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")
