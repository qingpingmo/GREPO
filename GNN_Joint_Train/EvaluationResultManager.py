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
from QueryEmbeddingGenerator import QueryEmbeddingGenerator
from QueryAwareTrainer import QueryAwareTrainer, create_nhop_subgraph_fast, create_inferer_filtered_subgraph, create_query_enhanced_features
from PrepareData import load_data, prepare_training_data_fast, get_test_split_boundary, prepare_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationResultManager:
    
    def __init__(self, repo_name: str, cache_dir: str = "./evaluation_cache", graph_data=None, inference_mode: bool = False, 
                 joint_training: bool = False, all_repo_names: List[str] = None):
        self.repo_name = repo_name
        self.cache_dir = cache_dir
        self.graph_data = graph_data
        self.inference_mode = inference_mode
        self.joint_training = joint_training
        self.all_repo_names = all_repo_names or [repo_name]
        os.makedirs(cache_dir, exist_ok=True)
        
        
        suffix = "_inference" if inference_mode else ""
        if joint_training:
            
            combined_repo_name = "_".join(sorted(self.all_repo_names))
            suffix += f"_joint_{combined_repo_name}"
        
    
        self.current_cache_file_train = os.path.join(cache_dir, f"{repo_name}_current_eval_cache_train{suffix}.json")
        self.current_cache_file_test = os.path.join(cache_dir, f"{repo_name}_current_eval_cache_test{suffix}.json")
        
        self.best_results_file = os.path.join(cache_dir, f"{repo_name}_best_results{suffix}.json")
        
        self.best_top10_score = 0.0
        
        self.node_id_to_info = {}
        self._build_node_mapping()
        
        self._load_best_results()
        
        mode_desc = "Joint training" if joint_training else "Single repo"
        logger.info(f"Result Manager Initialization - Mode: {mode_desc}, Repo: {repo_name}, Current Best Hit@10: {self.best_top10_score:.4f}")
    
    def _build_node_mapping(self):

        try:
            from datasets import Dataset
            import os
            
            repos_to_load = self.all_repo_names if self.joint_training else [self.repo_name]
            
            total_nodes_loaded = 0
            for repo_name in repos_to_load:
                logger.info(f"Construct node mapping for repo {repo_name} ...")
                
                dataset_paths = [
                    f"savedata/repos/{repo_name}/",
                    f"Graph_Feature_Construction/savedata/repos/{repo_name}/",
                    f"./savedata/repos/{repo_name}/"
                ]
                
                textgraphdataset = None
                for dataset_path in dataset_paths:
                    try:
                        if os.path.exists(dataset_path):
                            textgraphdataset = Dataset.load_from_disk(dataset_path)
                            logger.info(f"Successfully loaded the dataset from {dataset_path}")
                            break
                    except Exception as e:
                        continue
                
                if textgraphdataset is None:
                    logger.warning(f"Unable to find the dataset for {repo_name}, skip this repo")
                    continue
                
                repo_nodes_count = 0
                for item in textgraphdataset:
                    node_id = item.get("id")
                    if node_id is not None:
                        self.node_id_to_info[node_id] = {
                            "path": item.get("path", ""),
                            "name": item.get("name", ""),
                            "type": item.get("type", -1),
                            "start_commit": item.get("start_commit", ""),
                            "end_commit": item.get("end_commit", ""),
                            "repo_name": repo_name  
                        }
                        repo_nodes_count += 1
                
                logger.info(f"Repo {repo_name} 节点映射完成: {repo_nodes_count} 个节点")
                total_nodes_loaded += repo_nodes_count
            
            logger.info(f"Node mapping completed::  {total_nodes_loaded} nodes")
            
        except Exception as e:
            logger.warning(f"Node mapping failured: {e}")
            import traceback
            traceback.print_exc()
            self.node_id_to_info = {}
    
    def _convert_node_ids_to_info(self, node_ids: List[int]) -> List[Dict]:
        result = []
        missing_nodes = []
        
        for node_id in node_ids:
            node_info = self.node_id_to_info.get(node_id)
            if node_info:
                result.append({
                    "id": node_id,
                    "path": node_info["path"],
                    "name": node_info["name"],
                    "type": node_info["type"],
                    "repo_name": node_info.get("repo_name", "unknown") 
                })
            else:
                missing_nodes.append(node_id)
                result.append({
                    "id": node_id,
                    "path": f"unknown_path_for_id_{node_id}",
                    "name": f"unknown_name_for_id_{node_id}",
                    "type": -1,
                    "repo_name": "unknown"
                })
        
        if missing_nodes and len(missing_nodes) <= 10: 
            logger.warning(f"Mapping information for the following nodes could not be found: {missing_nodes[:10]}")
            logger.warning(f"The current mapping table contains {len(self.node_id_to_info)} nodes")
            if len(missing_nodes) > 10:
                logger.warning(f"another {len(missing_nodes) - 10} nodes missing...")
        
        return result
    
    def _load_best_results(self):
        return 
        if os.path.exists(self.best_results_file):
            try:
                with open(self.best_results_file, 'r', encoding='utf-8') as f:
                    best_data = json.load(f)
                    self.best_top10_score = best_data.get('best_top10_score', 0.0)
                    logger.info(f"best results loaded: hit10 = {self.best_top10_score:.4f}")
            except Exception as e:
                logger.warning(f"loading best results failured: {e}")
                self.best_top10_score = 0.0
        else:
            logger.info("No existing best result file found, starting from 0")
    
    def cache_evaluation_results(self, 
                               test_data: List,
                               predictions: Dict[str, List[int]], 
                               data_type: str = "test",
                               epoch: int = 0,
                               issue_idx: int = 0):

        current_results = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'issue_idx': issue_idx,
            'data_type': data_type,
            'repo_name': self.repo_name,
            'results': {}
        }
        
        for i, issue_data in enumerate(test_data):
            if len(issue_data) == 6:
                issue_idx_str, queries, anchor_labels, gt_labels, extra_info, repo_name = issue_data
            elif len(issue_data) >= 5:
                issue_idx_str, queries, anchor_labels, gt_labels, extra_info = issue_data
                repo_name = self.repo_name  
            else:
                issue_idx_str, queries, anchor_labels, gt_labels = issue_data
                extra_info = None
                repo_name = self.repo_name
            
            issue_result = {
                'issue_idx': issue_idx_str,
                'queries': queries,
                'data_type': data_type,
                'repo_name': repo_name  
            }
            
            if extra_info:
                gt_nodes_from_labels = torch.where(gt_labels > 0.5)[0].tolist() if gt_labels is not None else []
                
                issue_result.update({
                    'issue_time': extra_info.get('issue_time', -1),
                    'extractor_anchor_nodes': self._convert_node_ids_to_info(extra_info.get('extractor_nodes', [])),
                    'inferer_anchor_nodes': self._convert_node_ids_to_info(extra_info.get('inferer_nodes', [])),
                    'tarfiles_ids': self._convert_node_ids_to_info(gt_nodes_from_labels)  
                })
            else:
                gt_nodes_from_labels = torch.where(gt_labels > 0.5)[0].tolist() if gt_labels is not None else []
                issue_result.update({
                    'issue_time': -1,
                    'extractor_anchor_nodes': [],
                    'inferer_anchor_nodes': [],
                    'tarfiles_ids': self._convert_node_ids_to_info(gt_nodes_from_labels)
                })
            
            pred_key = str(issue_idx_str)
            if pred_key in predictions:
                issue_result.update({
                    'predicted_top10': self._convert_node_ids_to_info(predictions[pred_key].get('top10', [])),
                    'predicted_top20': self._convert_node_ids_to_info(predictions[pred_key].get('top20', []))
                })
            else:
                issue_result.update({
                    'predicted_top10': [],
                    'predicted_top20': []
                })
            
            current_results['results'][str(issue_idx_str)] = issue_result
        
        cache_file = self.current_cache_file_train if data_type == "train" else self.current_cache_file_test
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(current_results, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"cache {data_type} evaluation results: {len(current_results['results'])} issues")
    
    def update_best_results_if_better(self, current_top10_score: float, test_metrics: Dict[str, float]):

        if current_top10_score > self.best_top10_score:
            logger.info(f" Better results found! Hit10: {self.best_top10_score:.4f} -> {current_top10_score:.4f}")
            
            self.best_top10_score = current_top10_score
            
            all_cached_results = self._get_all_cached_results()
            
            if all_cached_results:

                best_results = {
                    'best_top10_score': current_top10_score,
                    'timestamp': datetime.now().isoformat(),
                    'epoch': all_cached_results.get('epoch', 0),
                    'issue_idx': all_cached_results.get('issue_idx', 0),
                    'repo_name': self.repo_name,
                    'test_metrics': test_metrics,
                    'results': all_cached_results.get('results', {})
                }
                

                with open(self.best_results_file, 'w', encoding='utf-8') as f:
                    json.dump(best_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Best results saved to: {self.best_results_file}")
                return True
            else:
                logger.warning("The current cache file does not exist, unable to save the best result")
                return False
        else:
            logger.debug(f"The current result does not exceed the best: {current_top10_score:.4f} <= {self.best_top10_score:.4f}")
            return False
    
    def _get_all_cached_results(self) -> Dict:

        all_results = {}
        
        train_cache_file = self.current_cache_file_train
        test_cache_file = self.current_cache_file_test
        
        if os.path.exists(test_cache_file):
            with open(test_cache_file, 'r', encoding='utf-8') as f:
                test_cache = json.load(f)
                all_results = {
                    'epoch': test_cache.get('epoch', 0),
                    'issue_idx': test_cache.get('issue_idx', 0),
                    'results': {}
                }
        
        if os.path.exists(test_cache_file):
            with open(test_cache_file, 'r', encoding='utf-8') as f:
                test_cache = json.load(f)
                for issue_id, result in test_cache.get('results', {}).items():
                    all_results['results'][issue_id] = result
        
        if os.path.exists(train_cache_file):
            with open(train_cache_file, 'r', encoding='utf-8') as f:
                train_cache = json.load(f)
                for issue_id, result in train_cache.get('results', {}).items():
                    all_results['results'][issue_id] = result
        
        return all_results
    
    def get_best_results(self) -> Dict:
        if os.path.exists(self.best_results_file):
            with open(self.best_results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {}
    
    def cleanup_cache(self):
        cache_files = [self.current_cache_file_train, self.current_cache_file_test]
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"Temporary cache files have been cleared: {cache_file}")

