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
from PrepareData import load_data, prepare_training_data_fast, get_test_split_boundary, prepare_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def save_checkpoint(model, optimizer, scheduler, epoch, issue_idx, best_f1, test_metrics, checkpoint_dir, filename_prefix="checkpoint", args=None):


    model_config = model.get_config() if hasattr(model, 'get_config') else {}
    repo_name = args.repo if args else "unknown_repo"
    
    subfolder_parts = [
        f"repo_{repo_name}",
        f"gnn_{model_config.get('gnn_type', 'unknown')}",
        f"layers_{model_config.get('num_layers', 'unknown')}",
        f"hidden_{model_config.get('hidden_dim', 'unknown')}",
        f"act_{model_config.get('activation', 'unknown')}",
        f"norm_{model_config.get('norm_type', 'none')}",
        f"drop_{model_config.get('dropout', 0.0)}"
    ]
    

    if model_config.get('use_residual', False):
        subfolder_parts.append(f"res_{model_config.get('residual_type', 'add')}")
    if model_config.get('num_heads', 1) > 1:
        subfolder_parts.append(f"heads_{model_config.get('num_heads')}")
    if model_config.get('mlp_layers', 1) > 1:
        subfolder_parts.append(f"mlp_{model_config.get('mlp_layers')}")
        
    subfolder_name = "_".join(subfolder_parts)
    
    full_checkpoint_dir = os.path.join(checkpoint_dir, subfolder_name)
    os.makedirs(full_checkpoint_dir, exist_ok=True)
    
    if args:
        hyperparams = vars(args).copy()  
        hyperparams['model_config'] = model_config
        hyperparams['timestamp'] = datetime.now().isoformat()
        
        hyperparams_path = os.path.join(full_checkpoint_dir, "hyperparameters.json")
        with open(hyperparams_path, 'w', encoding='utf-8') as f:
            json.dump(hyperparams, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"The hyperparameters have been saved to: {hyperparams_path}")
    

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'issue_idx': issue_idx,
        'best_f1': best_f1,
        'test_metrics': test_metrics,
        'model_config': model_config,
        'hyperparameters': vars(args) if args else {}
    }
    

    checkpoint_path = os.path.join(full_checkpoint_dir, f"{filename_prefix}_epoch{epoch}_issue{issue_idx}.pth")
    torch.save(checkpoint, checkpoint_path)
    

    latest_path = os.path.join(full_checkpoint_dir, "latest_checkpoint.txt")
    with open(latest_path, 'w') as f:
        f.write(checkpoint_path)
    
    logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info(f"Model configuration subfolder: {subfolder_name}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    resume_info = {
        'epoch': checkpoint.get('epoch', 0),
        'issue_idx': checkpoint.get('issue_idx', 0),
        'best_f1': checkpoint.get('best_f1', 0.0),
        'test_metrics': checkpoint.get('test_metrics', {}),
        'model_config': checkpoint.get('model_config', {})
    }
    
    logger.info(f"checkpoint - Epoch: {resume_info['epoch']}, Issue: {resume_info['issue_idx']}, Best F1: {resume_info['best_f1']:.4f}")
    return resume_info


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced RepoGNN with General MPNN Support')


    parser.add_argument("--repo", type=str, default="astropy")
    parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_hops', type=int, default=1, help='The hop count of the k-hop subgraph of extractor anchor nodes')
    parser.add_argument('--inferer_num_hops', type=int, default=1, help='The hop count of the k-hop subgraph of inferer anchor nodes')
    parser.add_argument('--max_subgraph_size', type=int, default=50000, help='Maximum number of nodes in the subgraph')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of training data samples')
    parser.add_argument('--eval_freq', type=int, default=50, 
                       help='Conduct an evaluation after training a certain number of issues')
    parser.add_argument('--eval_at_end', action='store_true', 
                       help='Evaluate at the end of each epoch')
    
    
    parser.add_argument('--input_dim', type=int, default=6, 
                       help='input_dim (num_queries + anchor_label)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='hidden_dim')
    parser.add_argument('--hidden_dim', type=int, default=1, help='hidden_dim')
    parser.add_argument('--num_layers', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='gcn', 
                       choices=['gcn', 'gat', 'gatv2', 'sage', 'gin', 'transformer', 
                               'graph', 'gated', 'arma', 'sg'],
                       help='GNN type')
    
    
    parser.add_argument('--activation', type=str, default='relu',
                       choices=['relu', 'gelu', 'elu', 'leaky_relu', 'selu', 
                               'swish', 'tanh', 'sigmoid', 'none'],
                       help='activation')
    parser.add_argument('--final_activation', type=str, default=None,
                       choices=['relu', 'gelu', 'elu', 'leaky_relu', 'selu', 
                               'swish', 'tanh', 'sigmoid', 'none'],
                       help='Final output activation function')
    parser.add_argument('--norm_type', type=str, default='none',
                       choices=['none', 'batch', 'layer', 'graph'],
                       help='norm_type')
    parser.add_argument('--norm_affine', action='store_true', help='If normalization learnable')
    
   
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--input_dropout', type=float, default=0.0, help='Input Dropout rate')
    parser.add_argument('--edge_dropout', type=float, default=0.0, help='Edge Dropout rate')
    parser.add_argument('--mlp_dropout', type=float, default=0.0, help='MLP Dropout rate')
    
    
    parser.add_argument('--use_residual', action='store_true', help='use residual connection or not')
    parser.add_argument('--residual_type', type=str, default='add',
                       choices=['add', 'concat'], help='residual_type')
    
    
    parser.add_argument('--num_heads', type=int, default=1, help='number of heads in attention')
    parser.add_argument('--attention_dropout', type=float, default=0.0, help='Attention Dropout rate')
    
    
    parser.add_argument('--aggr', type=str, default='add',
                       choices=['add', 'mean', 'max'], help='Aggregation Type')
    
    
    parser.add_argument('--layer_connection', type=str, default='sequential',
                       choices=['sequential', 'jumping_knowledge', 'dense'],
                       help='connection type between layers')
    parser.add_argument('--jk_mode', type=str, default='cat',
                       choices=['cat', 'max', 'lstm'], help='Jumping Knowledge mode')
    
    
    parser.add_argument('--mlp_layers', type=int, default=1, help='number of MLP layers')
    parser.add_argument('--mlp_hidden_dim', type=int, default=None, 
                       help='MLP hidden dimension (default is the same as hidden_dim)')
    
    
    parser.add_argument('--edge_dim', type=int, default=32, help='Edge feature dimension')
    
    
    parser.add_argument('--global_pool', type=str, default='none',
                       choices=['none', 'mean', 'max', 'add'], help='Global pooling type')
    
   
    parser.add_argument('--bias', action='store_true', default=True, help='bias')
    
    
    parser.add_argument('--resume_from', type=str, default="", 
                       help='Resume training from the specified checkpoint file')
    parser.add_argument('--save_checkpoint_freq', type=int, default=2000,
                       help='How often should a checkpoint be saved for each issue trained')
    parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints",
                       help='Checkpoint save directory')
    
    
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'step', 'cosine', 'none'], help='Learning Rate Scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='scheduler patience')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Scheduler decay factor')
    
    
    parser.add_argument('--loss_type', type=str, default='bce_with_logits',
                       choices=['bce_with_logits', 'focal', 'weighted_bce'], help='loss function type')
    parser.add_argument('--pos_weight', type=float, default=50.0, help='Positive sample weight')
    parser.add_argument('--focal_alpha', type=float, default=0.25, help='Focal loss alpha')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    
    
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient Clipping')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    
    
    parser.add_argument('--query_cache_file', type=str, default="query_embeddings_cache.pkl",
                       help='Query embedding cache file path')
    parser.add_argument('--evaluation_cache_dir', type=str, default="./evaluation_cache",
                       help='Evaluation results cache directory')
    
    
    parser.add_argument('--inference', action='store_true',
                       help='Enable inference mode and use the trained model for inference')
    parser.add_argument('--model_checkpoint', type=str, default="",
                       help='Model checkpoint path loaded in inference mode')
    parser.add_argument('--inference_rewriter_path', type=str, default="",
                       help='Rewriter file path used in inference mode')
    parser.add_argument('--inference_anchor_path', type=str, default="",
                       help='Path of the anchor file used in inference mode')
    
    return parser.parse_args()
