import pandas as pd
import os
import sys
import torch
import pickle
import json
import time
from datetime import datetime
from datasets import Dataset, load_from_disk
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
import logging
from collections import defaultdict
from debug_utils import SweProcessorDebugUtils

class SWEVerifiedProcessor(SweProcessorDebugUtils):
    def __init__(self, repognn_root="GREPO"):
        super().__init__()
        self.repognn_root = repognn_root
        self.repo_name_mapping = {
            'conda/conda': 'conda',
            'astropy/astropy': 'astropy', 
            'ipython/ipython': 'ipython',
            'google/jax': 'jax',
            'scikit-learn/scikit-learn': 'scikit-learn',
            'scipy/scipy': 'scipy',
            'huggingface/transformers': 'transformers',
            'django/django': 'django',
            
        }
        
        
        self.stats = {
            'total_instances': 0,
            'successful_instances': 0,
            'failed_instances': 0,
            'repo_stats': defaultdict(lambda: {'success': 0, 'failed': 0}),
            'processing_times': [],
            'error_counts': defaultdict(int),
            'function_counts': [],
            'adjacency_sizes': [],
            'start_time': None,
            'end_time': None
        }    
 
    def load_swe_verified_data(self, parquet_path: str) -> pd.DataFrame:
        
        self.logger.info(f"Loading SWE-Verified dataset from: {parquet_path}")
        
        if not os.path.exists(parquet_path):
            error_msg = f"Parquet file not found: {parquet_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            df = pd.read_parquet(parquet_path)
            self.stats['total_instances'] = len(df)
            
            self.logger.info(f"Loaded SWE-Verified dataset with {len(df)} instances")
            self.logger.info(f"Columns: {df.columns.tolist()}")
            
           
            repo_counts = df['repo'].value_counts()
            self.logger.info("Repository distribution:")
            for repo, count in repo_counts.items():
                self.logger.info(f"  {repo}: {count} instances")
            
            return df
            
        except Exception as e:
            error_msg = f"Error loading parquet file: {e}"
            self.logger.error(error_msg)
            raise
    
    def get_repo_sha_mapping(self, repo_name: str) -> Dict[str, int]:
        
        sha_path_file = os.path.join(self.repognn_root, "pulldata", f"{repo_name}_sha_path.pkl")
        
        self.logger.info(f"Loading SHA mapping for {repo_name}")
        
        if not os.path.exists(sha_path_file):
            error_msg = f"SHA path file not found for {repo_name}: {sha_path_file}"
            self.logger.warning(error_msg)
            self.stats['error_counts']['missing_sha_file'] += 1
            return {}
        
        try:
            with open(sha_path_file, "rb") as f:
                sha_path = pickle.load(f)
            
            
            sha_to_time = {sha: i for i, sha in enumerate(sha_path)}
            self.logger.info(f"Loaded {len(sha_to_time)} SHA mappings for {repo_name}")
            
            return sha_to_time
            
        except Exception as e:
            error_msg = f"Error loading SHA mapping for {repo_name}: {e}"
            self.logger.error(error_msg)
            self.stats['error_counts']['sha_loading_error'] += 1
            return {}
    
    def load_huggingface_dataset(self, repo_name: str) -> Optional[Dataset]:
        
        dataset_path = os.path.join(self.repognn_root, "savedata", "repos", repo_name)
        
        self.logger.info(f"Loading Huggingface dataset for {repo_name}")
        
        if not os.path.exists(dataset_path):
            error_msg = f"Dataset not found for {repo_name}: {dataset_path}"
            self.logger.warning(error_msg)
            self.stats['error_counts']['missing_dataset'] += 1
            return None
        
        try:
            dataset = load_from_disk(dataset_path)
            self.logger.info(f"Loaded dataset for {repo_name} with {len(dataset)} samples")
            
            self.logger.info(f"Dataset features: {list(dataset.features.keys())}")
            
            return dataset
            
        except Exception as e:
            error_msg = f"Error loading dataset for {repo_name}: {e}"
            self.logger.error(error_msg)
            self.stats['error_counts']['dataset_loading_error'] += 1
            return None
    
    def extract_subset_by_commit(self, dataset: Dataset, commit_sha: str, 
                                sha_to_time: Dict[str, int]) -> Optional[Dataset]:
        
        self.logger.info(f"Extracting subset for commit: {commit_sha}")
        
        if commit_sha not in sha_to_time:
            error_msg = f"Commit {commit_sha} not found in SHA mapping"
            self.logger.warning(error_msg)
            self.stats['error_counts']['commit_not_found'] += 1
            return None
        
        target_time = sha_to_time[commit_sha]
        self.logger.info(f"Target time: {target_time}")
        
        # starttime <= target_time < endtime
        def filter_by_time(example):
            start_time = example.get('start_commit', 'none')
            end_time = example.get('end_commit', 'none')
            
            
            start_timestamp = sha_to_time.get(start_time, sys.maxsize)
            end_timestamp = sha_to_time.get(end_time, sys.maxsize)
            
            return start_timestamp <= target_time < end_timestamp
        
        
        original_size = len(dataset)
        subset = dataset.filter(filter_by_time)
        filtered_size = len(subset)
        
        filter_ratio = filtered_size / original_size * 100 if original_size > 0 else 0
        self.logger.info(f"Filtered from {original_size} to {filtered_size} samples ({filter_ratio:.1f}%)")
        
        return subset
    


    def filter_functions_only(self, subset: Dataset) -> Dataset:
        
        self.logger.info("Filtering for function-level nodes only")
        
        def is_function_node(example):
            node_type = example.get('type', -1)
            
            # 0 = "directory", 1 = "file", 2 = "python file", 3 = "class def", 4 = "func def"
            
            return node_type in [2, 3, 4]  # python file, class def, func def
        
        original_size = len(subset)
        function_subset = subset.filter(is_function_node)
        filtered_size = len(function_subset)
        
        filter_ratio = filtered_size / original_size * 100 if original_size > 0 else 0
        self.logger.info(f"✓ Filtered to {filtered_size} function-level nodes ({filter_ratio:.1f}%)")
        
        
        type_counts = defaultdict(int)
        type_names = {0: "directory", 1: "file", 2: "python file", 3: "class def", 4: "func def"}
        
        for example in function_subset:
            node_type = example.get('type', -1)
            type_name = type_names.get(node_type, f"unknown({node_type})")
            type_counts[type_name] += 1
        
        self.logger.info("Node type distribution:")
        for node_type, count in type_counts.items():
            self.logger.info(f"  {node_type}: {count}")
        
        return function_subset

    def build_adjacency_list_and_code_dict(self, subset: Dataset) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        
        self.logger.info("Building adjacency list and code dictionary")
        
        # map: ID -> function_name
        id_to_function_name = {}
        function_name_to_code = {}
        
        self.logger.info("Phase 1: Building ID to function name mapping")
        
        
        for example in tqdm(subset, desc="Processing nodes"):
            node_id = example['id']
            function_name = self.create_function_name(example)
            id_to_function_name[node_id] = function_name
            
            
            node_type = example.get('type', -1)
            code = example.get('attr', '')
            
            
            type_names = {0: "directory", 1: "file", 2: "python file", 3: "class def", 4: "func def"}
            type_name = type_names.get(node_type, f"unknown({node_type})")
            
            if node_type == 4:  # func def
                
                if not code:
                    code = f"# Function definition for {function_name}"
            elif node_type == 3:  # class def
                
                if not code:
                    code = f"# Class definition for {function_name}"
            elif node_type == 2:  # python file
                
                if not code:
                    code = f"# Python file content for {function_name}"
            else:
                
                if not code:
                    code = f"# No code available for {function_name}"
            
            function_name_to_code[function_name] = code
        
        self.logger.info("Phase 2: Building adjacency relationships")
        
        # adj
        adjacency_list = {}
        neighbor_fields = ['call', 'contain', 'superclasses', 'previous']
        relationship_counts = defaultdict(int)
        
        for example in tqdm(subset, desc="Building adjacency"):
            node_id = example['id']
            function_name = id_to_function_name[node_id]
            
            
            if function_name not in adjacency_list:
                adjacency_list[function_name] = []
            
            
            for field in neighbor_fields:
                neighbors = example.get(field, [])
                if neighbors and isinstance(neighbors, (list, np.ndarray)):
                    for neighbor_id in neighbors:
                        if neighbor_id in id_to_function_name:
                            neighbor_function_name = id_to_function_name[neighbor_id]
                            
                            
                            if neighbor_function_name not in adjacency_list[function_name]:
                                adjacency_list[function_name].append(neighbor_function_name)
                                relationship_counts[field] += 1
                            
                            
                            if neighbor_function_name not in adjacency_list:
                                adjacency_list[neighbor_function_name] = []
                            if function_name not in adjacency_list[neighbor_function_name]:
                                adjacency_list[neighbor_function_name].append(function_name)
                                relationship_counts[f"{field}_reverse"] += 1
        
        
        total_nodes = len(adjacency_list)
        total_edges = sum(len(neighbors) for neighbors in adjacency_list.values())
        avg_degree = total_edges / total_nodes if total_nodes > 0 else 0
        
        self.logger.info(f"Built adjacency list with {total_nodes} nodes and {total_edges} edges")
        self.logger.info(f"Average degree: {avg_degree:.2f}")
        
        self.logger.info("Relationship type distribution:")
        for rel_type, count in relationship_counts.items():
            self.logger.info(f"  {rel_type}: {count}")
        
        
        type_distribution = defaultdict(int)
        type_names = {0: "directory", 1: "file", 2: "python file", 3: "class def", 4: "func def"}
        
        for example in subset:
            if example['id'] in id_to_function_name:
                node_type = example.get('type', -1)
                type_name = type_names.get(node_type, f"unknown({node_type})")
                type_distribution[type_name] += 1
        
        self.logger.info("Final node type distribution in adjacency list:")
        for node_type, count in type_distribution.items():
            self.logger.info(f"  {node_type}: {count}")
        
        
        self.stats['function_counts'].append(len(function_name_to_code))
        self.stats['adjacency_sizes'].append(total_edges)
        
        return adjacency_list, function_name_to_code  

def load_processed_instance(pickle_path: str) -> Dict[str, Any]:
    
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def main():
    
    parquet_path = "/path/to/verified-00000-of-00001.parquet"
    output_dir = "/path/to/output/"
    
   
    processor = SWEVerifiedProcessor()
    
    
    processor.process_all_instances(parquet_path, output_dir)
    
    
    processor.print_colorful("\n" + "="*60, "blue")
    processor.print_colorful("EXAMPLE USAGE", "green")
    processor.print_colorful("="*60, "blue")
    
    
    processed_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
    if processed_files:
        example_file = os.path.join(output_dir, processed_files[0])
        data = load_processed_instance(example_file)
        
        processor.print_colorful(f"Instance ID: {data['instance_id']}", "cyan")
        processor.print_colorful(f"Repository: {data['repo']}", "cyan")
        processor.print_colorful(f"Number of functions: {data['num_functions']}", "cyan")
        processor.print_colorful(f"Number of edges: {data['num_edges']}", "cyan")
        
        
        processor.print_colorful("\nAdjacency List Sample:", "yellow")
        for func_name, neighbors in list(data['adjacency_list'].items())[:3]:
            processor.print_colorful(f"  {func_name} -> {neighbors}", "white")
        
        
        processor.print_colorful("\nFunction Code Sample:", "yellow")
        for func_name, code in list(data['function_code_dict'].items())[:2]:
            processor.print_colorful(f"  {func_name}:", "white")
            processor.print_colorful(f"    {code[:100]}...", "white")

if __name__ == "__main__":
    main()