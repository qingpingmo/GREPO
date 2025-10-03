import optuna
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dev", type=int)
args = parser.parse_args()

device = args.dev

def obj(trial: optuna.Trial):
    hyperparams = {}
    hyperparams['gnn_type'] = trial.suggest_categorical('gnn_type', ['gat', 'gatv2', 'sage', 'gin'])
    
    hyperparams['hidden_dim'] = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256])
    
    hyperparams['num_layers'] = trial.suggest_int('num_layers', 1, 5)
    
    hyperparams['activation'] = "gelu"#trial.suggest_categorical('activation', ['relu', 'gelu', 'elu', 'leaky_relu', 'swish'])
    
    hyperparams['norm_type'] = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer', 'graph'])
    
    
    hyperparams['dropout'] = trial.suggest_float('dropout', 0.1, 0.6, step=0.1)
    hyperparams['input_dropout'] = trial.suggest_float('input_dropout', 0.0, 0.3, step=0.05)
    hyperparams['edge_dropout'] = trial.suggest_float('edge_dropout', 0.0, 0.3, step=0.05)
    
    
    hyperparams['use_residual'] = trial.suggest_categorical('use_residual', [True, False])
    if hyperparams['use_residual']:
        hyperparams['residual_type'] = trial.suggest_categorical('residual_type', ['add', 'concat'])
    else:
        hyperparams['residual_type'] = 'add'
    
    
    if hyperparams['gnn_type'] in ['gat', 'gatv2', 'transformer']:
        hyperparams['num_heads'] = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        hyperparams['attention_dropout'] = trial.suggest_float('attention_dropout', 0.0, 0.3, step=0.05)
    else:
        hyperparams['num_heads'] = 1
        hyperparams['attention_dropout'] = 0.0
    
    
    hyperparams['mlp_layers'] = trial.suggest_int('mlp_layers', 1, 3)
    hyperparams['mlp_dropout'] = trial.suggest_float('mlp_dropout', 0.0, 0.4, step=0.1)
    
    
    hyperparams['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    hyperparams['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    hyperparams['optimizer'] = "adamw"#trial.suggest_categorical('optimizer', ['adam', 'adamw'])
    
    
    hyperparams['loss_type'] = trial.suggest_categorical('loss_type', ['bce_with_logits', 'focal', 'weighted_bce'])
    hyperparams['pos_weight'] = trial.suggest_float('pos_weight', 10.0, 100.0, step=10.0)
    
    if hyperparams['loss_type'] == 'focal':
        hyperparams['focal_alpha'] = trial.suggest_float('focal_alpha', 0.1, 0.5, step=0.05)
        hyperparams['focal_gamma'] = trial.suggest_float('focal_gamma', 1.0, 3.0, step=0.25)
    else:
        hyperparams['focal_alpha'] = 0.25
        hyperparams['focal_gamma'] = 2.0
    
    
    hyperparams['scheduler'] = "none"#trial.suggest_categorical('scheduler', ['plateau', 'step', 'cosine', 'none'])
    if hyperparams['scheduler'] != 'none':
        hyperparams['scheduler_patience'] = trial.suggest_int('scheduler_patience', 3, 10)
        hyperparams['scheduler_factor'] = trial.suggest_float('scheduler_factor', 0.3, 0.8, step=0.1)
    else:
        hyperparams['scheduler_patience'] = 5
        hyperparams['scheduler_factor'] = 0.5
    
   
    hyperparams['num_hops'] = trial.suggest_int('num_hops', 1, 3)
    hyperparams['inferer_num_hops'] = trial.suggest_int('inferer_num_hops', 1, 3)
    #hyperparams['max_subgraph_size'] = trial.suggest_categorical('max_subgraph_size',  [20000, 30000, 40000, 50000, 60000])
    
    
    # hyperparams['grad_clip'] = trial.suggest_float('grad_clip', 0.5, 2.0, step=0.25)

    cmd = f"CUDA_VISIBLE_DEVICES={device} python GNN_Joint_Train/main.py --joint_training --repos astropy dvc ipython pylint scipy sphinx streamlink xarray geopandas --eval_freq -1 --epochs 1 --save_checkpoint_freq -1 --eval_at_end --checkpoint_dir ./checkpoints --query_cache_file query_embeddings.pkl --evaluation_cache_dir evaluation_cache --edge_dim 32 "
    for key in hyperparams:
        if isinstance(hyperparams[key], bool):
            if hyperparams[key]:
                cmd += f" --{key} " 
        else:
            cmd += f" --{key} " + str(hyperparams[key])
    out = subprocess.check_output(cmd, shell=True)
    out = out.decode()
    out = list(filter(lambda x: "Final Hit@10"in x, out.splitlines()))[0]
    print("test out", out.split()[-1], flush=True)
    return float(out.split()[-1])
