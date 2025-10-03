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
from ArgsUtil import save_checkpoint, load_checkpoint, parse_args 



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)








def run_inference(args):

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Inference Mode - Use Device: {device}")
    
    
    if not args.model_checkpoint:
        raise ValueError("In inference mode, --model_checkpoint must be specified")
    if not os.path.exists(args.model_checkpoint):
        raise FileNotFoundError(f"checkpoint not exists: {args.model_checkpoint}")
    
    logger.info(f"Inference mode - Load model checkpoint: {args.model_checkpoint}")
    
    
    rewriter_data, anchor_data, code_embeddings, graph_data, node_contents = load_data(args=args)
    
    
    datasets = prepare_training_data(rewriter_data, anchor_data, node_contents, graph_data, num_samples=args.num_samples)
    logger.info(f"XXXXXXXXXXXXXXXXXX{len(datasets)} instances ready XXXXXXXXXXXXXXXXXXXXXX")
    logger.info(f"Inference mode - A total of {len(datasets)} instances loaded, all used for infere")
    
    
    model = QueryAwareGNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        activation=args.activation,
        final_activation=args.final_activation,
        norm_type=args.norm_type,
        norm_affine=args.norm_affine,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        edge_dropout=args.edge_dropout,
        use_residual=args.use_residual,
        residual_type=args.residual_type,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        aggr=args.aggr,
        layer_connection=args.layer_connection,
        jk_mode=args.jk_mode,
        mlp_layers=args.mlp_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        edge_dim=args.edge_dim,
        global_pool=args.global_pool,
        bias=args.bias
    )
    model = model.to(device)
    
    
    checkpoint = torch.load(args.model_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Successfully loaded model weights - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    
    trainer = QueryAwareTrainer(model, device, args, query_cache_file=args.query_cache_file, graph_data=graph_data)
    graph_data = graph_data.to(device)
    
    logger.info("Starting inference evaluation...")
    inference_metrics = trainer.evaluate(
        datasets, code_embeddings, graph_data.edge_index, graph_data, 
        max_subgraph_size=args.max_subgraph_size,
        data_type="inference", epoch=0, issue_idx=0
    )
    
    
    logger.info(f"Inference complete!")
    logger.info(f"Hit@k:")
    for k in [1, 5, 10, 20]:
        logger.info(f"  Hit-{k}: {inference_metrics[f'gt_top_{k}_hit_rate']:.4f}")
    

    
    logger.info(f"The inference results have been saved to: {trainer.result_manager.current_cache_file_test}")
    
    trainer.query_generator.save_cache_final()
    
    logger.info("Inference mode completed")
    return inference_metrics


def main():
    args = parse_args()
    
    if args.joint_training and (not args.repos or len(args.repos) < 2):
        raise ValueError("In joint training mode, at least 2 repos must be specified (use --repos repo1 repo2 ...)")
    
    if args.inference:
        return run_inference(args)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    

    if args.joint_training:

        logger.info(f"Start joint training mode, repos: {args.repos}")
        
        multi_repo_data = load_data(args=args)
        

        joint_data = prepare_multi_repo_training_data(multi_repo_data, num_samples=args.num_samples, args=args)
        
        train_data = joint_data['combined_train_data']
        test_data = joint_data['combined_test_data']
        repo_embeddings = joint_data['repo_embeddings']
        repo_graphs = joint_data['repo_graphs']
        
        logger.info(f"Joint training data preparation completed:")
        logger.info(f"repos: {list(repo_embeddings.keys())}")
        logger.info(f"total training data: {len(train_data)} issues")
        logger.info(f"total test data: {len(test_data)} issues")
        
        
        for repo_name, repo_info in joint_data['repo_datasets'].items():
            train_count = len(repo_info['train'])
            test_count = len(repo_info['test'])
            logger.info(f"  {repo_name}: training{train_count}, test{test_count}")
        
        
        first_repo = list(repo_graphs.keys())[0]
        default_graph_data = repo_graphs[first_repo]
        code_embeddings = None  
        
    else:
        
        logger.info(f"Start single repo training mode, repo: {args.repo}")
        
        rewriter_data, anchor_data, code_embeddings, graph_data, node_contents = load_data(args=args)
        
        datasets = prepare_training_data(rewriter_data, anchor_data, node_contents, graph_data, num_samples=args.num_samples)
        
        logger.info("Starting dataset partitioning...")
        
        available_issue_ids = [int(issue_data[0]) for issue_data in datasets]
        test_boundary = get_test_split_boundary(args.repo, available_issue_ids)
        
        datasets.sort(key=lambda x: int(x[0]))
        
        train_data = []
        test_data = []
        
        for issue_data in datasets:
            issue_id = int(issue_data[0])
            if issue_id < test_boundary:
                train_data.append(issue_data)
            else:
                test_data.append(issue_data)
        
        logger.info(f"Single-repo Dataset division completed:")
        logger.info(f"  Train: {len(train_data)} issues")
        logger.info(f"  Test: {len(test_data)} issues")
        
       
        graph_data = graph_data.to(device)
        repo_embeddings = {args.repo: code_embeddings}
        repo_graphs = {args.repo: graph_data}
        default_graph_data = graph_data
    
    if len(train_data) == 0:
        logger.error("The training set is empty! Please check the data or boundary settings")
        return
    
    if len(test_data) == 0:
        logger.warning("The test set is empty! Please check the data or boundary settings")
    
    
    model = QueryAwareGNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        activation=args.activation,
        final_activation=args.final_activation,
        norm_type=args.norm_type,
        norm_affine=args.norm_affine,
        dropout=args.dropout,
        input_dropout=args.input_dropout,
        edge_dropout=args.edge_dropout,
        use_residual=args.use_residual,
        residual_type=args.residual_type,
        num_heads=args.num_heads,
        attention_dropout=args.attention_dropout,
        aggr=args.aggr,
        layer_connection=args.layer_connection,
        jk_mode=args.jk_mode,
        mlp_layers=args.mlp_layers,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_dropout=args.mlp_dropout,
        edge_dim=args.edge_dim,
        global_pool=args.global_pool,
        bias=args.bias
    ).to(device)
    
    print(model)
    logger.info(f"model config: {model.get_config()}")
    
    
    if args.joint_training:
        combined_repo_name = "_".join(sorted(args.repos))
        query_cache_file = f"query_embeddings_cache_joint_{combined_repo_name}.pkl"
        logger.info(f"Joint training mode uses query cache: {query_cache_file}")
    else:
        query_cache_file = args.query_cache_file
    
    
    # trainer = QueryAwareTrainer(
    #     model, device, args, 
    #     query_cache_file=query_cache_file, 
    #     graph_data=default_graph_data,
    #     repo_embeddings=repo_embeddings,
    #     repo_graphs=repo_graphs
    # )
    trainer = QueryAwareTrainer(
        model, device, args, 
        query_cache_file=query_cache_file, 
        graph_data=default_graph_data
    )
    
    
    start_epoch = 0
    start_issue_idx = 0
    best_f1 = 0
    
    
    if args.resume_from:
        try:
            resume_info = load_checkpoint(args.resume_from, model, trainer.optimizer, trainer.scheduler)
            start_epoch = resume_info['epoch']
            start_issue_idx = resume_info['issue_idx']
            best_f1 = resume_info['best_f1']
            logger.info(f"Resume training from checkpoint: Epoch {start_epoch}, Issue {start_issue_idx}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Train from scratch")
    
    num_epochs = args.epochs
    
    logger.info(f"Start training, a total of {num_epochs} epochs")
    logger.info(f"Evaluation frequency: Evaluate once after training every {args.eval_freq} issues")
    logger.info(f"Checkpoint save frequency: Save once every {args.save_checkpoint_freq} issues")
    
    # logger.info(f"before put graph memory {torch.cuda.max_memory_allocated()/1024**3}")
    # default_graph_data = default_graph_data.to(device)
    for reponame in repo_graphs:
        repo_graphs[reponame] = repo_graphs[reponame].to(device)
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        
        total_loss = 0
        num_batches = 0
        
        random.shuffle(train_data)
        
        start_idx = start_issue_idx if epoch == start_epoch else 0

        for i, issue_data in enumerate(tqdm(train_data[start_idx:], desc=f"Training Epoch {epoch+1}", initial=start_idx, total=len(train_data))):
            actual_i = start_idx + i

            if args.joint_training:
                
                issue_idx, queries, anchor_labels, gt_labels, extra_info, repo_name = issue_data
                code_embeddings = repo_embeddings[repo_name]
                current_graph_data = repo_graphs[repo_name]
                edge_index = current_graph_data.edge_index
                # logger.info(f"memory {torch.cuda.max_memory_allocated()/1024**3}")
                loss = trainer.train_on_issue(
                    issue_data, code_embeddings, edge_index, current_graph_data, max_subgraph_size=args.max_subgraph_size
                )
            else:
                
                loss = trainer.train_on_issue(
                    issue_data, code_embeddings, default_graph_data.edge_index, default_graph_data, 
                    max_subgraph_size=args.max_subgraph_size
                )
            total_loss += loss
            num_batches += 1
            
            
            if args.save_checkpoint_freq > 0 and (actual_i + 1) % args.save_checkpoint_freq == 0:
                save_checkpoint(
                    model, trainer.optimizer, trainer.scheduler, epoch, actual_i,
                    best_f1, {}, args.checkpoint_dir, "training_checkpoint", args
                )
            
            
            if args.eval_freq > 0 and (actual_i + 1) % args.eval_freq == 0:
                trainer.query_generator.save_cache_final()

                logger.info(f"\n--- Mid-term Evaluation (Epoch {epoch+1}, Issue {actual_i+1}/{len(train_data)}) ---")
                
                current_avg_loss = total_loss / num_batches
                logger.info(f"Current average training loss: {current_avg_loss:.4f}")
                

                if args.joint_training:
                    accumulated_gt_top_k_hits = {k: [] for k in [1, 5, 10, 20]}
                    accumulated_predictions = {}
                    accumulated_coverage_stats = {
                        'extractor_coverage': [],
                        'inferer_coverage': [],
                        'overall_coverage': [],
                        'extractor_nodes_count': [],
                        'inferer_nodes_count': [],
                        'combined_nodes_count': [],
                        'gt_nodes_count': []
                    }
                    

                    repo_specific_results = {}
                    for repo in args.repos:
                        repo_specific_results[repo] = {
                            'gt_top_k_hits': {k: [] for k in [1, 5, 10, 20]},
                            'predictions': {},
                            'coverage_stats': {
                                'extractor_coverage': [],
                                'inferer_coverage': [],
                                'overall_coverage': [],
                                'extractor_nodes_count': [],
                                'inferer_nodes_count': [],
                                'combined_nodes_count': [],
                                'gt_nodes_count': []
                            }
                        }
                    
                    with torch.no_grad():
                        for issue_data in tqdm(test_data, desc="Evaluating"):
                            issue_idx, queries, anchor_labels, gt_labels, extra_info, repo_name = issue_data
                            code_embeddings = repo_embeddings[repo_name]
                            current_graph_data = repo_graphs[repo_name]
                            edge_index = current_graph_data.edge_index
                                                        
                            gt_top_k_hits, predictions, coverage_stats = trainer.evaluate(
                                issue_data, code_embeddings, edge_index, current_graph_data, max_subgraph_size=args.max_subgraph_size,
                                data_type="test", epoch=epoch, issue_idx=issue_idx
                            )
                            

                            for k in [1, 5, 10, 20]:
                                accumulated_gt_top_k_hits[k].extend(gt_top_k_hits[k])
                            accumulated_predictions.update(predictions)
                            for key in accumulated_coverage_stats:
                                accumulated_coverage_stats[key].extend(coverage_stats[key])
                            

                            repo_acc = repo_specific_results[repo_name]
                            for k in [1, 5, 10, 20]:
                                repo_acc['gt_top_k_hits'][k].extend(gt_top_k_hits[k])
                            repo_acc['predictions'].update(predictions)
                            for key in repo_acc['coverage_stats']:
                                repo_acc['coverage_stats'][key].extend(coverage_stats[key])
                    

                    test_metrics = trainer.compute_results_and_manage_cache(
                        accumulated_gt_top_k_hits, test_data, accumulated_predictions, 
                        data_type="test", epoch=epoch, issue_idx=0, 
                        coverage_stats=accumulated_coverage_stats
                    )
                    

                    logger.info(f"\n--- Test results of each repository ---")
                    for repo_name, repo_results in repo_specific_results.items():
                        
                        repo_test_data = [data for data in test_data if data[-1] == repo_name]
                        
                        if not repo_test_data:  
                            continue
                            
                        repo_metrics = trainer.compute_results_and_manage_cache(
                            repo_results['gt_top_k_hits'], 
                            repo_test_data, 
                            repo_results['predictions'],
                            data_type="test", epoch=epoch, issue_idx=0,
                            coverage_stats=repo_results['coverage_stats']
                        )
                        
                        logger.info(f"repo '{repo_name}' results:")
                        for k in [1, 5, 10, 20]:
                            logger.info(f"  Test Top{k}: {repo_metrics[f'gt_top_{k}_hit_rate']:.4f}")
                        logger.info(f"  test issue: {len(repo_test_data)}")
                        
          
                else:
                   
                    test_metrics = trainer.evaluate(
                        test_data, code_embeddings, default_graph_data.edge_index, default_graph_data, 
                        max_subgraph_size=args.max_subgraph_size,
                        data_type="test", epoch=epoch, issue_idx=actual_i
                    )
                
                logger.info(f"Test - Hit@k:")
                for k in [1, 5, 10, 20]:
                    logger.info(f"  Test Top{k}: {test_metrics[f'gt_top_{k}_hit_rate']:.4f}")
                trainer.query_generator.save_cache_final()
 
                
                if test_metrics['gt_top_10_hit_rate'] > best_f1:
                    best_f1 = test_metrics['gt_top_10_hit_rate']
                    
                    save_checkpoint(
                        model, trainer.optimizer, trainer.scheduler, epoch, actual_i,
                        best_f1, test_metrics, args.checkpoint_dir, "best_model", args
                    )
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'epoch': epoch,
                        'issue_idx': actual_i,
                        'best_f1': best_f1,
                        'test_metrics': test_metrics,
                        'model_config': model.get_config()
                    }, "best_query_aware_model_enhanced.pth")
                    
                    logger.info(f"Best Hit10: {best_f1:.4f}")
                
                logger.info("--- Continue training ---\n")
            
            
            if num_batches % 20 == 0:
                torch.cuda.empty_cache()
        
        trainer.query_generator.save_cache_final()

       
        start_issue_idx = 0
        
        avg_loss = total_loss / max(num_batches, 1)
        
        
        if args.eval_at_end:
            logger.info(f"\n=== Epoch {epoch+1} End Evaluation ===")
            
       
            if args.joint_training:
                
                accumulated_gt_top_k_hits = {k: [] for k in [1, 5, 10, 20]}
                accumulated_predictions = {}
                accumulated_coverage_stats = {
                    'extractor_coverage': [],
                    'inferer_coverage': [],
                    'overall_coverage': [],
                    'extractor_nodes_count': [],
                    'inferer_nodes_count': [],
                    'combined_nodes_count': [],
                    'gt_nodes_count': []
                }
                
                
                repo_specific_results = {}
                for repo in args.repos:
                    repo_specific_results[repo] = {
                        'gt_top_k_hits': {k: [] for k in [1, 5, 10, 20]},
                        'predictions': {},
                        'coverage_stats': {
                            'extractor_coverage': [],
                            'inferer_coverage': [],
                            'overall_coverage': [],
                            'extractor_nodes_count': [],
                            'inferer_nodes_count': [],
                            'combined_nodes_count': [],
                            'gt_nodes_count': []
                        }
                    }
                
                with torch.no_grad():
                    for issue_data in tqdm(test_data, desc="Evaluating"):
                        issue_idx, queries, anchor_labels, gt_labels, extra_info, repo_name = issue_data
                        code_embeddings = repo_embeddings[repo_name]
                        current_graph_data = repo_graphs[repo_name]
                        edge_index = current_graph_data.edge_index
                                                    
                        gt_top_k_hits, predictions, coverage_stats = trainer.evaluate(
                            issue_data, code_embeddings, edge_index, current_graph_data, max_subgraph_size=args.max_subgraph_size,
                            data_type="test", epoch=epoch, issue_idx=len(train_data)-1
                        )
                        
                        
                        for k in [1, 5, 10, 20]:
                            accumulated_gt_top_k_hits[k].extend(gt_top_k_hits[k])
                        accumulated_predictions.update(predictions)
                        for key in accumulated_coverage_stats:
                            accumulated_coverage_stats[key].extend(coverage_stats[key])
                        
                        
                        repo_acc = repo_specific_results[repo_name]
                        for k in [1, 5, 10, 20]:
                            repo_acc['gt_top_k_hits'][k].extend(gt_top_k_hits[k])
                        repo_acc['predictions'].update(predictions)
                        for key in repo_acc['coverage_stats']:
                            repo_acc['coverage_stats'][key].extend(coverage_stats[key])
                
                
                test_metrics = trainer.compute_results_and_manage_cache(
                    accumulated_gt_top_k_hits, test_data, accumulated_predictions, 
                    data_type="test", epoch=epoch, issue_idx=0, 
                    coverage_stats=accumulated_coverage_stats
                )
                
                
                logger.info(f"\n--- Test results of each repository ---")
                for repo_name, repo_results in repo_specific_results.items():
                    
                    repo_test_data = [data for data in test_data if data[-1] == repo_name]
                    
                    if not repo_test_data:  
                        continue
                        
                    repo_metrics = trainer.compute_results_and_manage_cache(
                        repo_results['gt_top_k_hits'], 
                        repo_test_data, 
                        repo_results['predictions'],
                        data_type="test", epoch=epoch, issue_idx=0,
                        coverage_stats=repo_results['coverage_stats']
                    )
                    
                    logger.info(f"repo '{repo_name}' results:")
                    for k in [1, 5, 10, 20]:
                        logger.info(f"  Test Top{k}: {repo_metrics[f'gt_top_{k}_hit_rate']:.4f}")
                    logger.info(f"  Test issue: {len(repo_test_data)}")
                    
                     
            else:
                
                test_metrics = trainer.evaluate(
                    test_data, code_embeddings, default_graph_data.edge_index, default_graph_data, 
                    max_subgraph_size=args.max_subgraph_size,
                    data_type="test", epoch=epoch, issue_idx=len(train_data)-1
                )
            
            logger.info(f"Test - Hit@k:")
            for k in [1, 5, 10, 20]:
                logger.info(f"  Test Top{k}: {test_metrics[f'gt_top_{k}_hit_rate']:.4f}")           

            if test_metrics['gt_top_10_hit_rate'] > best_f1:
                best_f1 = test_metrics['gt_top_10_hit_rate']
                save_checkpoint(
                    model, trainer.optimizer, trainer.scheduler, epoch, len(train_data)-1,
                    best_f1, test_metrics, args.checkpoint_dir, "best_model", args
                )
                logger.info(f"Best Hit@10: {best_f1:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} finished - average training loss: {avg_loss:.4f}")
        
        
        save_checkpoint(
            model, trainer.optimizer, trainer.scheduler, epoch, len(train_data)-1,
            best_f1, {}, args.checkpoint_dir, "epoch_end", args
        )
        trainer.query_generator.save_cache_final()
        torch.cuda.empty_cache()
    
    
    trainer.query_generator.save_cache_final()
    
    
    trainer.result_manager.cleanup_cache()
    
    
    best_results = trainer.result_manager.get_best_results()
    if best_results:
        logger.info(f"   Final Best Results:")
        logger.info(f"   Hit@10: {best_results.get('best_top10_score', 0):.4f}")
        logger.info(f"   Results Saved to: {trainer.result_manager.best_results_file}")
        logger.info(f"   Detailed prediction results containing {len(best_results.get('results', {}))} issues")
    
    logger.info(f"Training Finished, best Hit@10: {best_f1:.4f}")
    print(f"Final Hit@10: {best_f1:.4f}", flush=True)
    logger.info(f"Query embedding cached saved to: {args.query_cache_file}")




if __name__ == "__main__":
    main()
