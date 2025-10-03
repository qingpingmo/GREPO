import argparse
import json
import logging
import os
import pickle
import random
import traceback
from typing import Dict, List, Tuple, Any
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import subgraph, k_hop_subgraph
from safetensors import safe_open
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from datasets import Dataset
from vllm import LLM
from transformers import AutoTokenizer
from QAGNN import QueryAwareGNN
from QueryEmbeddingGenerator import QueryEmbeddingGenerator
from EvaluationResultManager import EvaluationResultManager
from PrepareData import load_data, prepare_training_data_fast, get_test_split_boundary, prepare_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def create_nhop_subgraph_fast(anchor_nodes: torch.Tensor, 
                              edge_index: torch.Tensor,
                              num_nodes: int, 
                              num_hops: int = 2,
                              max_size: int = 8000) -> torch.Tensor:
    
    valid_anchor_nodes = anchor_nodes[0 <= anchor_nodes]

    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=valid_anchor_nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
        flow='source_to_target'
    )
    selected_nodes_list = subset
    return selected_nodes_list


def create_inferer_filtered_subgraph(inferer_nodes: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   node_types: torch.Tensor,
                                   num_nodes: int,
                                   num_hops: int = 2,
                                   max_size: int = 8000) -> torch.Tensor:

    valid_inferer_nodes = inferer_nodes[0 <= inferer_nodes]
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=valid_inferer_nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
        flow='source_to_target'
    )
    
    
    subset_types = node_types[subset]
    file_mask = (subset_types == 1) | (subset_types == 2)
    filtered_subset = subset[file_mask]
        
    return filtered_subset


def create_query_enhanced_features(node_embeddings: torch.Tensor, 
                                 query_embeddings: torch.Tensor) -> torch.Tensor:

    similarities = torch.matmul(node_embeddings, query_embeddings.T)  # [num_nodes, num_queries]
    # weights = F.softmax(similarities, dim=1)  # [num_nodes, num_queries]
    # weighted_query_features = torch.matmul(weights, query_embeddings)  # [num_nodes, query_dim]
    # enhanced_features = torch.cat([node_embeddings, weighted_query_features], dim=1)
    

    #return enhanced_features
    return similarities # [num_nodes, num_queries]

class QueryAwareTrainer:
    def __init__(self, model, device, args, query_cache_file="query_embeddings_cache.pkl", graph_data=None):
        self.model = model.to(device)
        self.device = device
        self.args = args
        self.num_hops = args.num_hops  
        self.inferer_num_hops = args.inferer_num_hops
        
        if not args.inference:
            self.optimizer = self._init_optimizer(model, args)
            
            self.criterion = self._init_criterion(args)
            
            self.scheduler = self._init_scheduler(self.optimizer, args)
            
            self.grad_accumulation_steps = args.grad_accumulation_steps
            self.grad_accumulation_counter = 0
        else:
            self.optimizer = None
            self.criterion = None
            self.scheduler = None
            self.grad_accumulation_steps = 1
            self.grad_accumulation_counter = 0
        
        self.query_generator = QueryEmbeddingGenerator(cache_file=query_cache_file)
        
        
        self.result_manager = EvaluationResultManager(
            repo_name=args.repo,
            cache_dir=getattr(args, 'evaluation_cache_dir', './evaluation_cache'),
            graph_data=graph_data,
            inference_mode=args.inference
        )
    
    def _init_optimizer(self, model, args):
        
        if args.optimizer.lower() == 'adam':
            return torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                                  weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'adamw':
            return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, 
                                   weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=args.learning_rate, 
                                 weight_decay=args.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    def _init_criterion(self, args):
        
        if args.loss_type == 'bce_with_logits':
            pos_weight = torch.tensor([args.pos_weight], device=self.device)
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif args.loss_type == 'weighted_bce':
            return nn.BCEWithLogitsLoss()
        elif args.loss_type == 'focal':
            return self._focal_loss
        else:
            raise ValueError(f"Unsupported loss type: {args.loss_type}")
    
    def _focal_loss(self, logits, targets):
        
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.args.focal_alpha * (1-pt)**self.args.focal_gamma * bce_loss
        return focal_loss.mean()
    
    def _init_scheduler(self, optimizer, args):
        
        if args.scheduler == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=args.scheduler_factor, 
                patience=args.scheduler_patience)
        elif args.scheduler == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.scheduler_patience, 
                gamma=args.scheduler_factor)
        elif args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs)
        elif args.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {args.scheduler}")
    
    def train_on_issue(self, issue_data, code_embeddings, edge_index, graph_data, max_subgraph_size=60000):
        
        self.model.train()
        
        if len(issue_data) == 6:  
            issue_idx, queries, anchor_labels, gt_labels, extra_info, repo_name = issue_data
            
            if True: #repo_name in self.repo_embeddings and repo_name in self.repo_graphs:
                #code_embeddings = self.repo_embeddings[repo_name]
                current_graph_data = graph_data
                edge_index = edge_index
            else:
                logger.warning(f"Data for the repo {repo_name} not found, using the provided default data")
                if code_embeddings is None or edge_index is None:
                    raise ValueError(f"Unable to find data for repo {repo_name}, and no default data provided")
        elif len(issue_data) == 5:  
            issue_idx, queries, anchor_labels, gt_labels, extra_info = issue_data
            repo_name = self.args.repo
            
            if code_embeddings is None or edge_index is None:
                raise ValueError("In single repo mode, the code_embeddings and edge_index parameters must be provided")
            current_graph_data = graph_data  
        else:
            raise ValueError(f"Unsupported issue_data format, length is{len(issue_data)}")

        logger.debug(f"Training issue {issue_idx} (repo: {repo_name})")
        
        #try:
        if True:
            query_embeddings = self.query_generator.generate_query_embeddings(queries)
            
            extractor_nodes = torch.tensor(extra_info['extractor_nodes']).to(self.device)
            extractor_nodes = create_nhop_subgraph_fast(
                    extractor_nodes, edge_index, code_embeddings.size(0), 
                    num_hops=self.num_hops, max_size=max_subgraph_size
                )
            
            inferer_nodes = torch.tensor(extra_info['inferer_nodes']).to(self.device)
            inferer_nodes = create_inferer_filtered_subgraph(
                    inferer_nodes, edge_index, graph_data.type, code_embeddings.size(0),
                    num_hops=self.inferer_num_hops, max_size=max_subgraph_size
                )
            
            all_nodes = torch.unique(torch.concat((inferer_nodes, extractor_nodes)))

            set_tensor = all_nodes
            node_embeddings = code_embeddings[set_tensor.to(code_embeddings.device)].to(self.device)
            
            enhanced_features = create_query_enhanced_features(
                node_embeddings.to(self.device), query_embeddings.to(self.device)
            )
            
            gt_labels = gt_labels[all_nodes.to(gt_labels.device)].to(self.device) 
            
            anchor_labels = torch.logical_or(torch.isin(all_nodes, extractor_nodes), torch.isin(all_nodes, inferer_nodes))

            anchor_features = anchor_labels.unsqueeze(-1)  
            enhanced_features_with_anchor = torch.cat([enhanced_features, anchor_features], dim=1)  
            
            edge_index = subgraph(nodes, edge_index, relabel_nodes=True)[0]
            
            logits = self.model(enhanced_features_with_anchor, edge_index)
            loss = self.criterion(logits, gt_labels)  
            
            loss = loss / self.grad_accumulation_steps
            loss.backward()
            
            self.grad_accumulation_counter += 1
            
            if self.grad_accumulation_counter % self.grad_accumulation_steps == 0:
 
                if self.args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            return loss.item() * self.grad_accumulation_steps  
            
        #except Exception as e:
        #    
        #    import traceback
        #    traceback.print_exc()
        #    return 0.0
    
    def _create_subgraph_edges_fast(self, edge_index, selected_nodes, node_mapping):
        subset_tensor = torch.tensor(selected_nodes, dtype=torch.long, device=edge_index.device)
        sub_edge_index, _ = subgraph(subset_tensor, edge_index, relabel_nodes=True, num_nodes=None)
        return sub_edge_index.to(self.device)
    
    def evaluate(self, test_data, code_embeddings, edge_index, graph_data, max_subgraph_size=60000, 
                 data_type="test", epoch=0, issue_idx=0):
        self.model.eval()
        gt_top_k_hits = {k: [] for k in [1, 5, 10, 20]}
        

        predictions = {}
        

        
        #with torch.no_grad():
        if True:
            #for issue_data in tqdm(test_data, desc="Evaluating"):
            if True:

                if len(test_data) == 6:  
                    current_issue_idx, queries, anchor_labels, gt_labels, extra_info, repo_name = test_data
                    if True: #repo_name in self.repo_embeddings and repo_name in self.repo_graphs:
                        # current_code_embeddings = self.repo_embeddings[repo_name]
                        # current_graph_data = self.repo_graphs[repo_name]
                        # current_edge_index = current_graph_data.edge_index
                        current_graph_data = graph_data
                        current_edge_index = edge_index
                        current_code_embeddings = code_embeddings
                    else:
                        logger.warning(f"Data for repo {repo_name} not found, skipping issue {current_issue_idx}")
                        return None, None, None
                elif len(test_data) == 5:  
                    current_issue_idx, queries, anchor_labels, gt_labels, extra_info = test_data
                    repo_name = self.args.repo
                    current_code_embeddings = code_embeddings
                    current_edge_index = edge_index
                    current_graph_data = graph_data
                    if current_code_embeddings is None or current_edge_index is None:
                        raise ValueError("In single repo mode, the code_embeddings and edge_index parameters must be provided")
                else:
                    logger.warning(f"Unsupported issue_data format, length is {len(test_data)}, skipping")
                    return None, None, None
                if True:
                    if gt_labels is None or gt_labels.sum() == 0:
                        return None, None, None
                    
                    query_embeddings = self.query_generator.generate_query_embeddings(queries)
                    
                    if extra_info:
                        extractor_nodes = torch.tensor(extra_info['extractor_nodes']).to(self.device)
                        
                        extractor_nodes = create_nhop_subgraph_fast(
                                extractor_nodes, current_edge_index, current_code_embeddings.get_shape()[0], 
                                num_hops=self.num_hops, max_size=max_subgraph_size
                            )


                        inferer_nodes = torch.tensor(extra_info['inferer_nodes']).to(self.device)
                        
                        inferer_nodes = create_inferer_filtered_subgraph(
                                inferer_nodes, current_edge_index, current_graph_data.type, current_code_embeddings.get_shape()[0],
                                num_hops=self.inferer_num_hops, max_size=max_subgraph_size
                            )

                        all_nodes = torch.unique(torch.concat((inferer_nodes, extractor_nodes)))
                        


                    
                    set_tensor = all_nodes.to(self.device) 
                    node_embeddings = current_code_embeddings[set_tensor.cpu()].to(self.device).float()
                    
                    enhanced_features = create_query_enhanced_features(
                        node_embeddings.to(self.device), query_embeddings.to(self.device)
                    )
                    

                    gt_labels = gt_labels.to(self.device)
                    gt_labels = gt_labels[all_nodes.to(gt_labels.device)].to(self.device) 
                    all_nodes = all_nodes.to(self.device)
                    
                    anchor_labels = torch.zeros(len(all_nodes), dtype=torch.float32)
                    if extra_info:
                        anchor_labels = torch.logical_or(torch.isin(all_nodes, torch.tensor(extra_info['extractor_nodes']).to(self.device)), torch.isin(all_nodes, torch.tensor(extra_info['inferer_nodes']).to(self.device)))
                    else:
                        anchor_labels = torch.logical_or(torch.isin(all_nodes, extractor_nodes), torch.isin(all_nodes, inferer_nodes))
                                

                    anchor_features = anchor_labels.unsqueeze(-1)  
                    enhanced_features_with_anchor = torch.cat([enhanced_features, anchor_features], dim=1)
                    
                    
                    current_edge_index = current_edge_index.to(self.device)
                    sub_edge_index, sub_edge_attr = subgraph(all_nodes, current_edge_index, current_graph_data.edge_attr, relabel_nodes=True)
                    logits = self.model(enhanced_features_with_anchor, sub_edge_index, sub_edge_attr)
                    preds = torch.sigmoid(logits)
                    
                    if len(preds) > 0:
                        _, top_indices = torch.topk(preds, min(20, len(preds)))
      
                        top10_ids = [all_nodes[idx].item() for idx in top_indices[:10].cpu().numpy()]
                        top20_ids = [all_nodes[idx].item() for idx in top_indices[:20].cpu().numpy()]
                        
                        predictions[str(current_issue_idx)] = {
                            'top10': top10_ids,
                            'top20': top20_ids,
                            'repo': repo_name
                        }
                    else:
                        predictions[str(current_issue_idx)] = {
                            'top10': [],
                            'top20': [],
                            'repo': repo_name
                        }
                    
                    if gt_labels.sum() > 0:
                        _, top_indices = torch.topk(preds, min(20, len(preds)))
                        true_gt_indices = torch.where(gt_labels > 0.5)[0]
                        
                        for k in [1, 5, 10, 20]:
                            if k <= len(preds):
                                hits = len(set(top_indices[:k].cpu().numpy()) & 
                                          set(true_gt_indices.cpu().numpy()))
                                hit_rate = hits / len(true_gt_indices) if len(true_gt_indices) > 0 else 0
                                gt_top_k_hits[k].append(hit_rate)
                                
         
                
                # except Exception as e:
                #     
                #     continue
                 
        return gt_top_k_hits, predictions, coverage_stats
    
    def step_scheduler(self, metric_value):
        
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metric_value)
            else:
                self.scheduler.step()