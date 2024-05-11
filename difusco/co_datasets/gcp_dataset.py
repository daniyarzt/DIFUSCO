"""GCP (Graph Coloring Problem) dataset."""

import glob
import os
import pickle

import numpy as np
import torch

from torch_geometric.data import Data as GraphData

class GCPDataset(torch.utils.data.Dataset):
  def __init__(self, data_file):
    self.data_file = data_file
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def get_example(self, idx):
    """
      Converts self.file_lines[idx] to num_nodes : int, node_labels : np.array, edges : np.array. 
    """ 
    # select sample
    line = self.file_lines[idx]

    # Clear leading/trailing characters
    line = line.strip()

    # get num_nodes and split edges
    edges, node_labels = line.split(' output ')
    num_nodes, edges = edges.split(' edges ')
    num_nodes = int(num_nodes)

    edges = edges.strip()
    edges = edges.split(' ') if edges != "" else []
    edges = np.array([[int(edges[i]), int(edges[i + 1])] for i in range(0, len(edges), 2)])

    edges = np.array(edges, dtype=np.int64)
    if edges.shape != (0, ):
      edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    # add self loop
    self_loop = np.arange(num_nodes).reshape(-1, 1).repeat(2, axis=1)
    if edges.shape != (0, ):
      edges = np.concatenate([edges, self_loop], axis=0)
    else:
      edges = self_loop
    edges = edges.T

    # get node labels 
    node_labels = node_labels.strip().split(' ')
    node_labels = np.array(node_labels, dtype=np.int64)

    return num_nodes, node_labels, edges

  def __getitem__(self, idx):
    num_nodes, node_labels, edge_index = self.get_example(idx)
    graph_data = GraphData(x=torch.from_numpy(node_labels),
                           edge_index=torch.from_numpy(edge_index))

    point_indicator = np.array([num_nodes], dtype=np.int64) # kept this from mis_dataset
    return (
        torch.LongTensor(np.array([idx], dtype=np.int64)), # kept this from mis_dataset
        graph_data,
        torch.from_numpy(point_indicator).long(), # kept this from mis_dataset
    )
