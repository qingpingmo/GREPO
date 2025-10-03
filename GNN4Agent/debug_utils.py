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
class SweProcessorDebugUtils:
    def __init__(self):
        self.setup_logging()   
    
    def setup_logging(self):
        
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"swe_processor_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logger initialized. Log file: {log_file}")
        
    def print_colorful(self, message: str, color: str = "white"):
        
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        
        color_code = colors.get(color, colors['white'])
        print(f"{color_code}{message}{colors['reset']}")
        
    def print_progress_bar(self, current: int, total: int, width: int = 50):
        
        progress = current / total
        filled_width = int(width * progress)
        bar = '█' * filled_width + '-' * (width - filled_width)
        percentage = progress * 100
        
        self.print_colorful(f"\rProgress: |{bar}| {percentage:.1f}% ({current}/{total})", "cyan")
       
    def create_function_name(self, example: Dict[str, Any]) -> str:
        
        path = example.get('path', 'unknown')
        name = example.get('name', 'unknown')
        node_type = example.get('type', -1)
        
        
        # 0 = "directory", 1 = "file", 2 = "python file", 3 = "class def", 4 = "func def"
        if node_type == 4:  # func def
            return f"{path}::{name}"
        elif node_type == 3:  # class def
            return f"{path}::{name}"
        elif node_type == 2:  # python file
            return f"{path}::{name}"
        elif node_type == 1:  # file
            return f"{path}::{name}"
        elif node_type == 0:  # directory
            return f"{path}::{name}"
        else:
            return f"{path}::{name}"    
        
    def print_statistics(self):
        
        self.print_colorful("\n" + "="*80, "blue")
        self.print_colorful("PROCESSING STATISTICS", "green")
        self.print_colorful("="*80, "blue")
        
       
        total = self.stats['total_instances']
        success = self.stats['successful_instances']
        failed = self.stats['failed_instances']
        
        self.print_colorful(f"Total Instances: {total}", "white")
        if total > 0:
            self.print_colorful(f"Successful: {success} ({success/total*100:.1f}%)", "green")
            self.print_colorful(f"Failed: {failed} ({failed/total*100:.1f}%)", "red")
        
        
        if self.stats['processing_times']:
            avg_time = np.mean(self.stats['processing_times'])
            total_time = sum(self.stats['processing_times'])
            self.print_colorful(f"Average Processing Time: {avg_time:.2f}s", "cyan")
            self.print_colorful(f"Total Processing Time: {total_time:.2f}s", "cyan")
        
        
        if self.stats['function_counts']:
            avg_functions = np.mean(self.stats['function_counts'])
            total_functions = sum(self.stats['function_counts'])
            max_functions = max(self.stats['function_counts'])
            min_functions = min(self.stats['function_counts'])
            self.print_colorful(f"Average Nodes per Instance: {avg_functions:.1f}", "yellow")
            self.print_colorful(f"Max Nodes per Instance: {max_functions}", "yellow")
            self.print_colorful(f"Min Nodes per Instance: {min_functions}", "yellow")
            self.print_colorful(f"Total Nodes Processed: {total_functions}", "yellow")
        
        
        if self.stats['adjacency_sizes']:
            avg_edges = np.mean(self.stats['adjacency_sizes'])
            total_edges = sum(self.stats['adjacency_sizes'])
            max_edges = max(self.stats['adjacency_sizes'])
            min_edges = min(self.stats['adjacency_sizes'])
            self.print_colorful(f"Average Edges per Instance: {avg_edges:.1f}", "magenta")
            self.print_colorful(f"Max Edges per Instance: {max_edges}", "magenta")
            self.print_colorful(f"Min Edges per Instance: {min_edges}", "magenta")
            self.print_colorful(f"Total Edges Processed: {total_edges}", "magenta")
        
        
        self.print_colorful("\nRepository Statistics:", "magenta")
        for repo, stats in self.stats['repo_stats'].items():
            total_repo = stats['success'] + stats['failed']
            success_rate = stats['success'] / total_repo * 100 if total_repo > 0 else 0
            self.print_colorful(f"  {repo}: {stats['success']}/{total_repo} ({success_rate:.1f}%)", "white")
        
        
        if self.stats['error_counts']:
            self.print_colorful("\nError Statistics:", "red")
            for error_type, count in self.stats['error_counts'].items():
                self.print_colorful(f"  {error_type}: {count}", "white")

    
    def save_final_statistics(self, output_dir: str):
        
        stats_file = os.path.join(output_dir, "processing_statistics.json")
        
        
        serializable_stats = {
            'total_instances': self.stats['total_instances'],
            'successful_instances': self.stats['successful_instances'],
            'failed_instances': self.stats['failed_instances'],
            'repo_stats': dict(self.stats['repo_stats']),
            'error_counts': dict(self.stats['error_counts']),
            'processing_times': {
                'average': np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'total': sum(self.stats['processing_times']),
                'min': min(self.stats['processing_times']) if self.stats['processing_times'] else 0,
                'max': max(self.stats['processing_times']) if self.stats['processing_times'] else 0,
            },
            'function_counts': {
                'average': np.mean(self.stats['function_counts']) if self.stats['function_counts'] else 0,
                'total': sum(self.stats['function_counts']),
                'min': min(self.stats['function_counts']) if self.stats['function_counts'] else 0,
                'max': max(self.stats['function_counts']) if self.stats['function_counts'] else 0,
            },
            'start_time': self.stats['start_time'],
            'end_time': self.stats['end_time'],
            'total_duration': self.stats['end_time'] - self.stats['start_time'] if self.stats['end_time'] and self.stats['start_time'] else 0
        }
        
        with open(stats_file, 'w') as f:
            json.dump(serializable_stats, f, indent=2)
        
        self.logger.info(f"Final statistics saved to {stats_file}")
     
    def process_single_instance(self, instance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        
        instance_start_time = time.time()
        
        repo_full_name = instance['repo']
        base_commit = instance['base_commit']
        instance_id = instance['instance_id']
        
        self.print_colorful(f"\n{'='*80}", "blue")
        self.print_colorful(f"Processing Instance: {instance_id}", "green")
        self.print_colorful(f"Repository: {repo_full_name}", "yellow")
        self.print_colorful(f"Base Commit: {base_commit}", "yellow")
        self.print_colorful(f"{'='*80}", "blue")
        
        self.logger.info(f"Processing instance {instance_id} for repo {repo_full_name}")
        
        
        # if repo_full_name not in self.repo_name_mapping:
        #     error_msg = f"Repo {repo_full_name} not in mapping"
        #     self.logger.warning(error_msg)
        #     self.stats['error_counts']['unmapped_repo'] += 1
        #     self.stats['repo_stats'][repo_full_name]['failed'] += 1
        #     return None
        
        #repo_name = self.repo_name_mapping[repo_full_name]
        repo_name = repo_full_name.split('/')[-1]
        
        try:
           
            self.print_colorful("Step 1: Loading SHA mapping...", "cyan")
            sha_to_time = self.get_repo_sha_mapping(repo_name)
            if not sha_to_time:
                self.stats['repo_stats'][repo_full_name]['failed'] += 1
                return None
            
            
            self.print_colorful("Step 2: Loading Huggingface dataset...", "cyan")
            dataset = self.load_huggingface_dataset(repo_name)
            if dataset is None:
                self.stats['repo_stats'][repo_full_name]['failed'] += 1
                return None
            
            
            self.print_colorful("Step 3: Extracting subset by commit...", "cyan")
            subset = self.extract_subset_by_commit(dataset, base_commit, sha_to_time)
            if subset is None:
                self.stats['repo_stats'][repo_full_name]['failed'] += 1
                return None
            
            
            self.print_colorful("Step 4: Filtering function-level nodes...", "cyan")
            function_subset = self.filter_functions_only(subset)
            if len(function_subset) == 0:
                self.logger.warning(f"No function nodes found for instance {instance_id}")
                self.stats['error_counts']['no_functions'] += 1
                self.stats['repo_stats'][repo_full_name]['failed'] += 1
                return None
            
            
            self.print_colorful("Step 5: Building adjacency list and code dictionary...", "cyan")
            adjacency_list, code_dict = self.build_adjacency_list_and_code_dict(function_subset)
            
            
            processing_time = time.time() - instance_start_time
            self.stats['processing_times'].append(processing_time)
            
            
            result = {
                'instance_id': instance_id,
                'repo': repo_full_name,
                'base_commit': base_commit,
                'adjacency_list': adjacency_list,
                'function_code_dict': code_dict,
                'num_functions': len(code_dict),
                'num_edges': sum(len(neighbors) for neighbors in adjacency_list.values()),
                'processing_time': processing_time,
                'problem_statement': instance.get('problem_statement', ''),
                'patch': instance.get('patch', ''),
                'test_patch': instance.get('test_patch', ''),
                'processed_at': datetime.now().isoformat()
            }
            
            self.print_colorful(f"✓ Successfully processed {instance_id}", "green")
            self.print_colorful(f"  Functions: {len(code_dict)}", "green")
            self.print_colorful(f"  Edges: {result['num_edges']}", "green")
            self.print_colorful(f"  Processing time: {processing_time:.2f}s", "green")
            
            self.stats['repo_stats'][repo_full_name]['success'] += 1
            return result
            
        except Exception as e:
            error_msg = f"Error processing instance {instance_id}: {e}"
            self.logger.error(error_msg)
            self.stats['error_counts']['processing_error'] += 1
            self.stats['repo_stats'][repo_full_name]['failed'] += 1
            return None
    
    def save_processed_data(self, processed_data: Dict[str, Any], output_path: str):
        
        self.logger.info(f"Saving processed data to {output_path}")
        
        try:
            
            with open(output_path, 'wb') as f:
                pickle.dump(processed_data, f)
            
            
            summary_path = output_path.replace('.pkl', '_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(f"Instance Summary\n")
                f.write(f"{'='*50}\n")
                f.write(f"Instance ID: {processed_data['instance_id']}\n")
                f.write(f"Repository: {processed_data['repo']}\n")
                f.write(f"Base Commit: {processed_data['base_commit']}\n")
                f.write(f"Number of Functions: {processed_data['num_functions']}\n")
                f.write(f"Number of Edges: {processed_data['num_edges']}\n")
                f.write(f"Processing Time: {processed_data['processing_time']:.2f}s\n")
                f.write(f"Processed At: {processed_data['processed_at']}\n")
                f.write(f"\nFunction Names:\n")
                for i, func_name in enumerate(processed_data['function_code_dict'].keys()):
                    f.write(f"  {i+1}. {func_name}\n")
                f.write(f"\nAdjacency List Sample (first 10):\n")
                for i, (func_name, neighbors) in enumerate(list(processed_data['adjacency_list'].items())[:10]):
                    f.write(f"  {func_name} -> {neighbors}\n")
                    
            self.logger.info(f"✓ Data saved successfully")
            
        except Exception as e:
            error_msg = f"Error saving data: {e}"
            self.logger.error(error_msg)
            self.stats['error_counts']['save_error'] += 1
    
   
    def process_all_instances(self, parquet_path: str, output_dir: str = "processed_swe_verified"):
        
        self.stats['start_time'] = time.time()
        
        
        swe_df = self.load_swe_verified_data(parquet_path)
        
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")
        
        self.print_colorful(f"\nStarting processing of {len(swe_df)} instances", "green")
        self.print_colorful(f"Output directory: {output_dir}", "cyan")
        
       
        with tqdm(total=len(swe_df), desc="Processing instances", unit="instance") as pbar:
            for idx, row in swe_df.iterrows():
                instance_id = row['instance_id']
                
                if os.path.exists(os.path.join(output_dir, f"{instance_id}.pkl")):
                    self.logger.info(f"Skipping {instance_id}, already processed")
                    self.stats['successful_instances'] += 1
                    pbar.update(1)
                    continue
                processed_data = self.process_single_instance(row.to_dict())
                
                if processed_data is not None:
                    
                    output_path = os.path.join(output_dir, f"{instance_id}.pkl")
                    self.save_processed_data(processed_data, output_path)
                    
                    self.stats['successful_instances'] += 1
                else:
                    self.stats['failed_instances'] += 1
                
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': self.stats['successful_instances'],
                    'Failed': self.stats['failed_instances'],
                    'Success Rate': f"{self.stats['successful_instances']/(idx+1)*100:.1f}%"
                })
                
                
                if (idx + 1) % 10 == 0:
                    self.print_statistics()
        
        self.stats['end_time'] = time.time()
        
        
        self.print_statistics()
        
        
        self.save_final_statistics(output_dir)
        
        self.print_colorful(f"\nProcessing complete! Results saved to {output_dir}", "green")

