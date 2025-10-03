import argparse
import json
import logging
import os
import pickle
import random
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, GINConv, GINEConv, TransformerConv, 
    GATv2Conv, GraphConv, GatedGraphConv, ARMAConv, SGConv, GPSConv,
    global_mean_pool, global_max_pool, global_add_pool,
    LayerNorm, BatchNorm, GraphNorm
)
from torch_geometric.utils import subgraph, k_hop_subgraph
from safetensors import safe_open
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from datasets import Dataset
from vllm import LLM
from transformers import AutoTokenizer
from PrepareData import load_data, prepare_training_data_fast, get_test_split_boundary, prepare_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralGNNLayer(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 gnn_type: str = 'gcn',
                 num_heads: int = 1,
                 dropout: float = 0.0,
                 bias: bool = True,
                 edge_dim: Optional[int] = None,
                 aggr: str = 'add',
                 **kwargs):
        super().__init__()
        
        self.gnn_type = gnn_type.lower()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if self.gnn_type == 'gcn':
            self.conv = GCNConv(input_dim, output_dim, bias=bias, **kwargs)
        elif self.gnn_type == 'gat':
            
            per_head_dim = output_dim // num_heads
            if output_dim % num_heads != 0:
                per_head_dim += 1
                actual_output_dim = per_head_dim * num_heads
            else:
                actual_output_dim = output_dim
            self.conv = GATConv(input_dim, per_head_dim, heads=num_heads, 
                               dropout=dropout, bias=bias, edge_dim=edge_dim, concat=True, **kwargs)
            
            if actual_output_dim != output_dim:
                self.output_proj = nn.Linear(actual_output_dim, output_dim, bias=bias)
            else:
                self.output_proj = nn.Identity()
        elif self.gnn_type == 'gatv2':
            
            per_head_dim = output_dim // num_heads
            if output_dim % num_heads != 0:
                per_head_dim += 1
                actual_output_dim = per_head_dim * num_heads
            else:
                actual_output_dim = output_dim
            self.conv = GATv2Conv(input_dim, per_head_dim, heads=num_heads,
                                 dropout=dropout, bias=bias, edge_dim=edge_dim, concat=True, **kwargs)
            if actual_output_dim != output_dim:
                self.output_proj = nn.Linear(actual_output_dim, output_dim, bias=bias)
            else:
                self.output_proj = nn.Identity()
        elif self.gnn_type == 'sage':
            mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            self.conv = GINEConv(mlp, edge_dim=edge_dim, aggr="mean", **kwargs)
        elif self.gnn_type == 'gin':
            mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            self.conv = GINEConv(mlp, edge_dim=edge_dim, **kwargs) #GINConv(mlp, aggr=aggr, **kwargs)
        elif self.gnn_type == 'transformer':
            self.conv = TransformerConv(input_dim, output_dim, heads=num_heads,
                                       dropout=dropout, edge_dim=edge_dim, bias=bias, **kwargs)
        elif self.gnn_type == 'graph':
            self.conv = GraphConv(input_dim, output_dim, aggr=aggr, bias=bias, **kwargs)
        elif self.gnn_type == 'gated':
            self.conv = GatedGraphConv(output_dim, num_layers=1, aggr=aggr, bias=bias, **kwargs)
        elif self.gnn_type == 'arma':
            self.conv = ARMAConv(input_dim, output_dim, bias=bias, **kwargs)
        elif self.gnn_type == 'sg':
            self.conv = SGConv(input_dim, output_dim, bias=bias, **kwargs)
        elif self.gnn_type == 'gps':
            mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            )
            localconv = GINEConv(mlp, edge_dim=edge_dim, aggr="mean", **kwargs)
            self.conv = GPSConv(input_dim, localconv, heads=num_heads)
            self.output_proj = nn.Linear(input_dim, output_dim, bias=bias)
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
    
    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        if True: # self.gnn_type in ['gat', 'gatv2', 'transformer'] and edge_attr is not None:
            out = self.conv(x, edge_index, edge_attr=edge_attr, **kwargs)
        else:
            out = self.conv(x, edge_index, **kwargs)
        
        
        if hasattr(self, 'output_proj'):
            out = self.output_proj(out)
            
        return out


class QueryAwareGNN(nn.Module):
    
    def __init__(self, 
                 input_dim: int = 6,
                 hidden_dim: int = 256,
                 output_dim: int = 1,
                 num_layers: int = 2,
                 gnn_type: str = 'gcn',
                 
                 # Activation
                 activation: str = 'relu',
                 final_activation: Optional[str] = None,
                 
                 # Normalization
                 norm_type: str = 'none',  # none, batch, layer, graph
                 norm_affine: bool = True,
                 
                 # Dropout
                 dropout: float = 0.3,
                 input_dropout: float = 0.0,
                 edge_dropout: float = 0.0,
                 
                 # Residual
                 use_residual: bool = False,
                 residual_type: str = 'add',  # add, concat
                 
                 # Attention
                 num_heads: int = 1,
                 attention_dropout: float = 0.0,
                 
                 # Aggregation
                 aggr: str = 'add',  # add, mean, max
                 
                 # connection
                 layer_connection: str = 'sequential',  # sequential, jumping_knowledge, dense
                 jk_mode: str = 'cat',  # cat, max, lstm
                 
                 # MLP
                 mlp_layers: int = 1,
                 mlp_hidden_dim: Optional[int] = None,
                 mlp_dropout: float = 0.0,
                 
                 # edge feature
                 edge_dim: Optional[int] = None,
                 
                 # pool
                 global_pool: str = 'none',  # none, mean, max, add
                 
                 # others
                 bias: bool = True,
                 **kwargs):
        super().__init__()
        print("edge_dim!", edge_dim)
        self.edge_emb = nn.Embedding(100, edge_dim)

        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.activation = activation.lower()
        self.final_activation = final_activation.lower() if final_activation else None
        self.norm_type = norm_type.lower()
        self.use_residual = use_residual
        self.residual_type = residual_type.lower()
        self.layer_connection = layer_connection.lower()
        self.jk_mode = jk_mode.lower()
        self.global_pool = global_pool.lower()
        
        
        self.act_fn = self._get_activation_fn(activation)
        self.final_act_fn = self._get_activation_fn(final_activation) if final_activation else None
        
        
        self.input_proj = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()  
        
        for i in range(num_layers):
            
            layer_input_dim = hidden_dim
            layer_output_dim = hidden_dim
            
            if layer_connection == 'dense' and i > 0:
                layer_input_dim = hidden_dim * (i + 1)
            
            
            gnn_layer = GeneralGNNLayer(
                input_dim=layer_input_dim,
                output_dim=layer_output_dim,
                gnn_type=gnn_type,
                num_heads=num_heads,
                dropout=attention_dropout,
                bias=bias,
                edge_dim=edge_dim,
                aggr=aggr,
                **kwargs
            )
            self.gnn_layers.append(gnn_layer)
            
            
            if use_residual and residual_type == 'concat':
                
                self.residual_projections.append(
                    nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
                )
            else:
                self.residual_projections.append(nn.Identity())
            
            
            if norm_type != 'none':
                if norm_type == 'batch':
                    norm_layer = BatchNorm(layer_output_dim, affine=norm_affine)
                elif norm_type == 'layer':
                    norm_layer = LayerNorm(layer_output_dim, affine=norm_affine)
                elif norm_type == 'graph':
                    norm_layer = GraphNorm(layer_output_dim)
                else:
                    raise ValueError(f"Unsupported norm type: {norm_type}")
                self.norm_layers.append(norm_layer)
            else:
                self.norm_layers.append(nn.Identity())
            
            
            self.dropout_layers.append(nn.Dropout(dropout))
        
        # Jumping Knowledge
        if layer_connection == 'jumping_knowledge':
            if jk_mode == 'cat':
                self.jk_input_dim = hidden_dim * num_layers
            elif jk_mode == 'lstm':
                self.jk_lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.jk_input_dim = hidden_dim
            else:  # max
                self.jk_input_dim = hidden_dim
        else:
            self.jk_input_dim = hidden_dim
        
        
        mlp_hidden_dim = mlp_hidden_dim or hidden_dim
        self.output_mlp = self._build_mlp(
            input_dim=self.jk_input_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=output_dim,
            num_layers=mlp_layers,
            dropout=mlp_dropout,
            bias=bias
        )
        
        
        if global_pool == 'mean':
            self.global_pool_fn = global_mean_pool
        elif global_pool == 'max':
            self.global_pool_fn = global_max_pool
        elif global_pool == 'add':
            self.global_pool_fn = global_add_pool
        else:
            self.global_pool_fn = None
    
    def _get_activation_fn(self, activation: str):
        
        if activation == 'relu':
            return F.relu
        elif activation == 'gelu':
            return F.gelu
        elif activation == 'elu':
            return F.elu
        elif activation == 'leaky_relu':
            return F.leaky_relu
        elif activation == 'selu':
            return F.selu
        elif activation == 'swish':
            return F.silu
        elif activation == 'tanh':
            return torch.tanh
        elif activation == 'sigmoid':
            return torch.sigmoid
        elif activation == 'none' or activation is None:
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int, 
                   num_layers: int, dropout: float, bias: bool):
        
        if num_layers == 1:
            return nn.Linear(input_dim, output_dim, bias=bias)
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim, bias=bias))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim, bias=bias))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, edge_index, edge_attr, batch=None, **kwargs):
        
        assert edge_attr is not None
        edge_attr = self.edge_emb(edge_attr)
        
        h = self.input_proj(x)
        h = self.input_dropout(h)
        
        
        layer_outputs = []
        
        
        for i, (gnn_layer, norm_layer, dropout_layer, residual_proj) in enumerate(
            zip(self.gnn_layers, self.norm_layers, self.dropout_layers, self.residual_projections)):
            
            
            if self.layer_connection == 'dense' and i > 0:
                h = torch.cat(layer_outputs + [h], dim=-1)
            
            residual = h if self.use_residual else None
            
            h = gnn_layer(h, edge_index, edge_attr, **kwargs)
            
            
            if self.norm_type == 'graph' and batch is not None:
                h = norm_layer(h, batch)
            else:
                h = norm_layer(h)
            
            h = self.act_fn(h)
            
            if residual is not None:
                if self.residual_type == 'add':
                    
                    if residual.size(-1) == h.size(-1):
                        h = h + residual
                if self.residual_type == 'concat':
                    h = torch.cat([h, residual], dim=-1)
                    
                    h = nn.Linear(h.size(-1), self.hidden_dim, device=h.device, bias=True)(h)
            
            h = dropout_layer(h)
            
            layer_outputs.append(h)
        
        # Jumping Knowledge
        if self.layer_connection == 'jumping_knowledge':
            if self.jk_mode == 'cat':
                h = torch.cat(layer_outputs, dim=-1)
            elif self.jk_mode == 'max':
                h = torch.stack(layer_outputs, dim=-1).max(dim=-1)[0]
            elif self.jk_mode == 'lstm':
                
                seq = torch.stack(layer_outputs, dim=1)  # [num_nodes, num_layers, hidden_dim]
                _, (h, _) = self.jk_lstm(seq)
                h = h.squeeze(0)  # [num_nodes, hidden_dim]

        if self.global_pool_fn is not None and batch is not None:
            h = self.global_pool_fn(h, batch)
        
        logits = self.output_mlp(h)
        
        if self.final_act_fn is not None:
            logits = self.final_act_fn(logits)
        
        if self.output_dim == 1:
            logits = logits.squeeze(-1)
        
        return logits
    
    def get_config(self):
       
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'gnn_type': self.gnn_type,
            'activation': self.activation,
            'final_activation': self.final_activation,
            'norm_type': self.norm_type,
            'use_residual': self.use_residual,
            'residual_type': self.residual_type,
            'layer_connection': self.layer_connection,
            'jk_mode': self.jk_mode,
            'global_pool': self.global_pool,
        }