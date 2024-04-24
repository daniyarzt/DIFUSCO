import argparse
import pathlib
import glob
import networkx as nx
import pickle

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Simple MIS solver")
  parser.add_argument("input_folder", type=str, action="store", help="Directory containing input")

  args = parser.parse_args()
  print(args.input_folder)

  files = glob.glob(args.input_folder + '/' + '*.gpickle')
  for file in files: 
    with open(file, 'rb') as f:
      g = pickle.load(f)
    mis = nx.maximal_independent_set(g)
    for node, attrs in g.nodes(data=True):
      attrs['label'] = int(node in mis)

    with open(file, 'wb') as f:
      pickle.dump(g, f)

  print('Done! Labels are added :)')