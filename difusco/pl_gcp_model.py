"""Lightning module for training the DIFUSCO GCP Model"""
import os

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from co_datasets.gcp_dataset import GCPDataset
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.gcp_utils import gcp_decode_np
from utils.gcp_utils import count_gcp_violations

class GCPModel(COMetaModel):
  def __init__(self, param_args=None):
    super(GCPModel, self).__init__(param_args=param_args, node_feature_only=True)

    self.num_colors = self.args.K
    self.train_dataset = GCPDataset(
      data_file = os.path.join(self.args.storage_path, self.args.training_split),
    )

    self.test_dataset = GCPDataset(
      data_file = os.path.join(self.args.storage_path, self.args.test_split),
    )

    self.validation_dataset = GCPDataset(
      data_file = os.path.join(self.args.storage_path, self.args.validation_split),
    )

  def forward(self, x, t, edge_index):
    return self.model(x, t, edge_index = edge_index)

  def categorical_training_step(self, batch, batch_idx):
    _, graph_data, point_indicator = batch
    t = np.random.randint(1, self.diffusion.T + 1, point_indicator.shape[0]).astype(int)
    node_labels = graph_data.x
    edge_index = graph_data.edge_index

    # Sample from diffusion 
    node_labels_onehot = F.one_hot(node_labels.long(), num_classes=self.num_colors).float()
    node_labels_onehot = node_labels_onehot.unsqueeze(1).unsqueeze(1)

    t = torch.from_numpy(t).long() 
    t = t.repeat_interleave(point_indicator.reshape(-1).cpu(), dim = 0).numpy()

    xt = self.diffusion.sample(node_labels_onehot, t)
    xt = xt * 2 - 1 # ?? 
    xt * (1.0 * 0.05 * torch.rand_like(xt))

    t = torch.from_numpy(t).float() 
    t = t.reshape(-1) 
    xt = xt.reshape(-1) 
    edge_index = edge_index.to(node_labels.device).reshape(2, -1) # ??

    x0_pred = self.forward(
      xt.float().to(node_labels.device), 
      t.float().to(node_labels.device), 
      edge_index
    )

    loss_func = nn.CrossEntropyLoss() 
    loss = loss_func(x0_pred, node_labels) # ?? 
    self.log("train/loss", loss)
    return loss

  def gaussian_training_step(self, batch, batch_idx):
    raise NotImplementedError()

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      x0_pred = self.forward(
        xt.float().to(device), 
        t.float().to(device), 
        edge_index.long().to(device) if edge_index is not None else None, 
      )
      x0_pred_prob = x0_pred.reshape((1, xt.shape[0], -1, self.num_colors)).softmax(dim = -1)
      xt = self.new_categorical_posterior(target_t, t, x0_pred_prob, xt.to(device))
      return xt

  def gaussian_denoise_step(self, xt, t, device, edge_index=None, target_t=None):
    raise NotImplementedError()

  def test_step(self, batch, batch_idx, draw=False, split='test'):
    if self.diffusion_type == 'gaussian':
      raise NotImplemented('test_step() is only implemented for Categorical Diffusion.')
    device = batch[-1].device

    real_batch_idx, graph_data, point_indicator = batch 
    node_labels = graph_data.x
    edge_index = graph_data.edge_index 

    stacked_predict_labels = []
    edge_index = edge_index.to(node_labels.device).reshape(2, -1)
    edge_index_np = edge_index.cpu().numpy() 

    adj_mat = scipy.sparse.coo_matrix(
      (np.ones_like(edge_index_np[0]), (edge_index_np[0], edge_index_np[1]))
    )

    for _ in range(self.args.sequential_sampling): 
      xt = torch.randint(low = 0, high = self.num_colors, size = node_labels.shape).long()
      if self.args.parallel_sampling > 1: # ?
        xt = xt.repeat(self.args.parallel_sampling, 1, 1)
        xt = torch.randint(low = 0, high = self.num_colors, size = xt.shape).long()
       
      xt.reshape(-1) # ?

      if self.args.parallel_sampling > 1: 
        edge_index = self.duplicate_edge_index(edge_index, node_labels.shape[0], device)

      batch_size = 1
      steps = self.args.inference_diffusion_steps
      time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule, 
                                        T=self.diffusion.T, inference_T=steps)
      
      # Inference process
      for i in range(steps):
        t1, t2 = time_schedule(i)
        t1 = np.array([t1 for _ in range(batch_size)]).astype(int)
        t2 = np.array([t2 for _ in range(batch_size)]).astype(int)
                
        xt = self.categorical_denoise_step(
          xt, t1, device, edge_index, target_t=t2
        )        
      
      predict_labels = xt.float().cpu().detach().numpy() + 1e-6 # ?
      stacked_predict_labels.append(predict_labels)

    predict_labels = np.concatenate(stacked_predict_labels, axis = 0)
    all_sampling = self.args.sequential_sampling * self.args.parallel_sampling

    splitted_predict_labels = np.split(predict_labels, all_sampling)
    solved_solutions = [gcp_decode_np(predict_labels, adj_mat) for  predict_labels in splitted_predict_labels]
    solved_costs = [count_gcp_violations(solved_solution, adj_mat) for solved_solution in solved_solutions]
    best_solved_cost = np.max(solved_costs)

    gt_cost = node_labels.cpu().numpy().sum() 
    metrics = {
      f"{split}/gt_cost": gt_cost
    }
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
    self.log(f"{split}/solved_cost", best_solved_cost, prog_bar=True, on_epoch=True, sync_dist=True)

  def validation_step(self, batch, batch_idx):
    return self.test_step(batch, batch_idx, split='val')