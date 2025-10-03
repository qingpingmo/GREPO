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
from QueryAwareTrainer import QueryAwareTrainer, create_nhop_subgraph_fast, create_inferer_filtered_subgraph, create_query_enhanced_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_single_repo_data(repo_name, inference_mode=False, inference_rewriter_path=None, inference_anchor_path=None):

    logger.info(f"Loading data for repo {repo_name}...")
    
    if inference_mode:
        
        logger.info(f"Inference mode - Loading rewriter data: {inference_rewriter_path}")
        with open(inference_rewriter_path, 'r') as f:
            rewriter_data = json.load(f)
        
        logger.info(f"Inference Mode - Loading Anchor Data: {inference_anchor_path}")
        with open(inference_anchor_path, 'r') as f:
            anchor_data = json.load(f)
    else:
        
        with open(f'Graph_Feature_Construction/{repo_name}/rewriter_output_post.json', 'r') as f:
            rewriter_data = json.load(f)
        
        with open(f'Graph_Feature_Construction/{repo_name}/anchor_node.json', 'r') as f:
            anchor_data = json.load(f)
    
    
    if os.path.exists(f'Graph_Feature_Construction/get_content_embedding/output_{repo_name}/{repo_name}_embeddings.safetensors'):
        with safe_open(f'Graph_Feature_Construction/get_content_embedding/output_{repo_name}/{repo_name}_embeddings.safetensors', framework="pt") as f:
            embedding_matrix = f.get_slice('embeddings')
    else:
        with safe_open(f'Graph_Feature_Construction/get_content_embedding/{repo_name}/{repo_name}_embeddings.safetensors', framework="pt") as f:
            embedding_matrix = f.get_slice('embeddings')    
    # graph
    graph_data = torch.load(f'pyggraph/{repo_name}.timed.pt', weights_only=False)

    
    node_contents = {}
    
    logger.info(f"Repo {repo_name} - Rewriter: {len(rewriter_data)} ")
    logger.info(f"Repo {repo_name} - Anchor: {len(anchor_data)} ")
    logger.info(f"Repo {repo_name} - code embeddings: {embedding_matrix.get_shape()}")
    logger.info(f"Repo {repo_name} - graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
    
    return rewriter_data, anchor_data, embedding_matrix, graph_data, node_contents


def load_data(args=None):

    logger.info("Loading data... ")

    if args.inference:
        if not args.inference_rewriter_path or not args.inference_anchor_path:
            raise ValueError("In inference mode, you must specify --inference_rewriter_path and --inference_anchor_path")
        
       
        return load_single_repo_data(
            repo_name=args.repo,
            inference_mode=True,
            inference_rewriter_path=args.inference_rewriter_path,
            inference_anchor_path=args.inference_anchor_path
        )
    
    
    if args.joint_training:
        if not args.repos or len(args.repos) < 2:
            raise ValueError("In joint training mode, at least 2 repos must be specified (use --repos repo1 repo2 ...)")
        
        logger.info(f"Joint Training Mode - Loading {len(args.repos)} repos: {args.repos}")
        
        multi_repo_data = {}
        for repo_name in args.repos:
            try:
                repo_data = load_single_repo_data(repo_name, inference_mode=False)
                multi_repo_data[repo_name] = repo_data
                logger.info(f"Successfully loaded repo: {repo_name}")
            except Exception as e:
                logger.error(f"Failed to load repo {repo_name}: {e}")
                raise ValueError(f"Unable to load repo {repo_name}: {e}")
        
        logger.info(f"Joint training data loading completed, with {len(multi_repo_data)} repos")
        return multi_repo_data
    
    else:
        
        logger.info(f"Single repo training mode - Load repo: {args.repo}")
        return load_single_repo_data(args.repo, inference_mode=False)


def prepare_training_data_fast(rewriter_data, anchor_data, node_contents, graph_data, num_samples=100):
    
    num_nodes = graph_data.num_nodes
    max_node_id = num_nodes - 1
    
    
    valid_samples = []
    logger.info("Preprocessing...")
    
    
    for idx_str, anchor_item in anchor_data.items():
        rewriter_item = rewriter_data.get(idx_str)
        if not rewriter_item:
            continue
            
        
        queries = rewriter_item.get('query')
        extractor_anchor_nodes = anchor_item.get('extractor_anchor_nodes')
        inferer_anchor_nodes = anchor_item.get('inferer_anchor_nodes')
        gt_nodes = anchor_item.get('tarfiles_ids')
        

        combined_anchor_nodes = []
        if extractor_anchor_nodes:
            combined_anchor_nodes.extend(extractor_anchor_nodes)
        if inferer_anchor_nodes:
            
            for node_list in inferer_anchor_nodes:
                combined_anchor_nodes.extend(node_list)
        
        combined_anchor_nodes = list(set(combined_anchor_nodes))
        
        if queries and combined_anchor_nodes and gt_nodes:
            
            valid_samples.append((int(idx_str), queries, combined_anchor_nodes, gt_nodes, 
                                extractor_anchor_nodes or [], inferer_anchor_nodes or []))
    
    logger.info(f"Preprocessing complete: {len(valid_samples)} valid samples")
    datasets = []
    total_gt_matches = 0
    torch_zeros = torch.zeros
    
    for i, (idx, queries, combined_anchor_nodes, gt_nodes, extractor_anchors, inferer_anchors) in enumerate(valid_samples):
        
        
        combined_anchor_array = np.array(combined_anchor_nodes, dtype=np.int32)
        gt_array = np.array(gt_nodes, dtype=np.int32)
        combined_anchor_mask = (combined_anchor_array >= 0) & (combined_anchor_array <= max_node_id)
        gt_mask = (gt_array >= 0) & (gt_array <= max_node_id)
        valid_combined_anchors = combined_anchor_array[combined_anchor_mask]
        valid_gts = gt_array[gt_mask]
        
        
        intersection = np.intersect1d(valid_gts, valid_combined_anchors)
        overlap_ratio = len(intersection) / len(valid_gts) if len(valid_gts) > 0 else 0.0
        
        
        extractor_array = np.array(extractor_anchors, dtype=np.int32) if extractor_anchors else np.array([], dtype=np.int32)
        inferer_flat = []
        if inferer_anchors:
            for node_list in inferer_anchors:
                inferer_flat.extend(node_list)
        inferer_array = np.array(list(set(inferer_flat)), dtype=np.int32) if inferer_flat else np.array([], dtype=np.int32)
        
        
        valid_extractors = extractor_array[(extractor_array >= 0) & (extractor_array <= max_node_id)] if len(extractor_array) > 0 else np.array([], dtype=np.int32)
        valid_inferers = inferer_array[(inferer_array >= 0) & (inferer_array <= max_node_id)] if len(inferer_array) > 0 else np.array([], dtype=np.int32)
        
        if len(valid_combined_anchors) > 0 and len(valid_gts) > 0:
            anchor_labels = torch_zeros(num_nodes, dtype=torch.float32)
            gt_labels = torch_zeros(num_nodes, dtype=torch.float32)
            
            anchor_labels[valid_combined_anchors] = 1.0
            gt_labels[valid_gts] = 1.0
            
           
            extra_info = {
                'extractor_nodes': valid_extractors.tolist(),
                'inferer_nodes': valid_inferers.tolist(),
                'combined_nodes': valid_combined_anchors.tolist()
            }
            
            datasets.append((idx, queries, anchor_labels, gt_labels, extra_info))
            total_gt_matches += len(valid_gts)
            
            if i < 2 or (i + 1) % 500 == 0:
                logger.info(f"{i+1}/{len(valid_samples)} processed")
    
    logger.info(f"processing finished！{len(datasets)} instances，{total_gt_matches} GT nodes")
    return datasets


def get_test_split_boundary(repo_name: str, available_issue_ids: List[int]) -> int:

    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return int(max(available_issue_ids) * 0.8)
    
    parquet_path = "verified-00000-of-00001.parquet"
    
    try:
        
        df = pd.read_parquet(parquet_path)
        logger.info(f"Successfully read the parquet file, containing {len(df)} records")
        
        if 'repo' not in df.columns:
            logger.warning("There is no 'repo' column in the parquet file, using the default split ratio of 0.8")
            return int(max(available_issue_ids) * 0.8)
        
        repo_df = df[df['repo'] == repo_name]
        
        if len(repo_df) == 0:
            matching_repos = df[df['repo'].str.contains(repo_name, case=False, na=False)]['repo'].unique()
            if len(matching_repos) == 0:
                logger.warning(f"No matching data for repo '{repo_name}' was found in the parquet file, using the d")
                return int(max(available_issue_ids) * 0.8)
            elif len(matching_repos) > 1:
                logger.warning(f"Found multiple matching repos: {matching_repos}, use the first one: {matching_repos[0]}")
                repo_df = df[df['repo'] == matching_repos[0]]
            else:
                logger.info(f"matched repo found: '{repo_name}' -> '{matching_repos[0]}'")
                repo_df = df[df['repo'] == matching_repos[0]]
        
        logger.info(f"Found {len(repo_df)} records for the repo '{repo_name}'")
        
        parquet_issue_ids = set()
        

        if 'issue_numbers' in repo_df.columns:
            for issue_numbers in repo_df['issue_numbers']:
                if isinstance(issue_numbers, (list, tuple)):
                    for issue_num in issue_numbers:
                        try:
                            parquet_issue_ids.add(int(issue_num))
                        except (ValueError, TypeError):
                            continue
                else:
                    try:
                        parquet_issue_ids.add(int(issue_numbers))
                    except (ValueError, TypeError):
                        continue
        else:

            if 'instance_id' in repo_df.columns:
                for instance_id in repo_df['instance_id']:
                    try:
                        
                        issue_num = instance_id.split('-')[-1]
                        parquet_issue_ids.add(int(issue_num))
                    except (ValueError, TypeError):
                        continue
        
        logger.info(f"Extracted {len(parquet_issue_ids)} issue_ids from the parquet file")
        
        
        available_issue_set = set(available_issue_ids)
        intersection = parquet_issue_ids.intersection(available_issue_set)
        
        logger.info(f"There are currently {len(available_issue_set)} issue_ids")
        logger.info(f"There are {len(intersection)} issue_ids in the intersection with the parquet file")
        
        if len(intersection) == 0:
            logger.warning("There is no intersection between the current data and the parquet file, using the default split ratio of 0.8.")
            return int(max(available_issue_ids) * 0.8)
        
        min_test_issue_id = min(intersection)
        
        logger.info(f"train: issue_id < {min_test_issue_id}")
        logger.info(f"test: issue_id >= {min_test_issue_id}")
        
        return min_test_issue_id
        
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        logger.info("Use the default split ratio of 0.8")
        return int(max(available_issue_ids) * 0.8)


def prepare_multi_repo_training_data(multi_repo_data, num_samples=100, args=None):

    logger.info(f"Preparing multi-repo joint training data, a total of {len(multi_repo_data)} repos")
    
    repo_datasets = {}
    repo_embeddings = {}
    repo_graphs = {}
    combined_train_data = []
    combined_test_data = []
    
    for repo_name, (rewriter_data, anchor_data, embedding_matrix, graph_data, node_contents) in multi_repo_data.items():
        logger.info(f"Processing repo: {repo_name}")
        
        datasets = prepare_training_data_fast(rewriter_data, anchor_data, node_contents, graph_data, num_samples)
        
        if not datasets:
            logger.warning(f"Repo {repo_name} has no valid data, skipping")
            continue
        
        available_issue_ids = [int(issue_data[0]) for issue_data in datasets]
        test_boundary = get_test_split_boundary(repo_name, available_issue_ids)
        
        datasets.sort(key=lambda x: int(x[0]))
        
        repo_train_data = []
        repo_test_data = []
        
        for issue_data in datasets:
            issue_id = int(issue_data[0])

            enhanced_issue_data = list(issue_data) + [repo_name]  
            
            if issue_id < test_boundary:
                repo_train_data.append(enhanced_issue_data)
            else:
                repo_test_data.append(enhanced_issue_data)
        
        logger.info(f"Repo {repo_name} - Training data: {len(repo_train_data)}, Test data: {len(repo_test_data)}")
        
        repo_datasets[repo_name] = {
            'train': repo_train_data,
            'test': repo_test_data
        }
        repo_embeddings[repo_name] = embedding_matrix
        repo_graphs[repo_name] = graph_data
        
        combined_train_data.extend(repo_train_data)
        combined_test_data.extend(repo_test_data)
    
    logger.info(f"Joint training data preparation completed:")
    logger.info(f"  Total training data: {len(combined_train_data)} issues")
    logger.info(f"  Total test data: {len(combined_test_data)} issues")
    
    random.shuffle(combined_train_data)
    
    return {
        'combined_train_data': combined_train_data,
        'combined_test_data': combined_test_data,
        'repo_datasets': repo_datasets,
        'repo_embeddings': repo_embeddings,
        'repo_graphs': repo_graphs
    }



def prepare_training_data(rewriter_data, anchor_data, node_contents, graph_data, num_samples=100):
    return prepare_training_data_fast(rewriter_data, anchor_data, node_contents, graph_data, num_samples)