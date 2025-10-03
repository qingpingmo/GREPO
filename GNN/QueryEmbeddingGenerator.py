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
from EvaluationResultManager import EvaluationResultManager
from QueryAwareTrainer import QueryAwareTrainer, create_nhop_subgraph_fast, create_inferer_filtered_subgraph, create_query_enhanced_features
from PrepareData import load_data, prepare_training_data_fast, get_test_split_boundary, prepare_training_data


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEmbeddingGenerator:
    def __init__(self, cache_file="query_embeddings_cache.pkl"):
        self.model = None
        self.tokenizer = None
        self.max_length = 32768
        self.cache_file = cache_file
        self.cache = self._load_cache()
        logger.info(f"Query embedding cache initialization complete, {len(self.cache)} queries cache")
    
    def get_model(self, ):
        if self.model is None:
            self.model = LLM(
                model="Qwen3-Embedding-8B",
                task="embed",
                enforce_eager=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                'Qwen3-Embedding-8B',
                padding_side='left',
                trust_remote_code=True,
                local_files_only=False
            )

    def _load_cache(self):
        
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"Successfully loaded query embedding cache: {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, using empty cache")
                return {}
        else:
            logger.info("Cache file does not exist, using empty cache")
            return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Cache saved: {len(self.cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        return query
    
    def generate_query_embeddings(self, queries: List[str]) -> torch.Tensor:

        if not queries:
            return torch.zeros(0, 4096)
        
        ret = []
        for query in queries:
            if query in self.cache:
                logger.debug(f"Retrieve query embeddings from cache (cache key: {query}...)")
                ret.append(self.cache[query])
                continue
            
            logger.debug(f"generate new query embeddings (cache key: {query}...)")

            self.get_model()        
            prompt_token_ids = self.tokenizer(
                [query],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            
            prompts = self.tokenizer.batch_decode(
                prompt_token_ids["input_ids"], 
                skip_special_tokens=True
            )
            
            emb_outputs = self.model.embed(prompts)
            
            if hasattr(emb_outputs[0], "embedding"):
                embeddings = [out.embedding for out in emb_outputs]
            elif hasattr(emb_outputs[0], "outputs") and hasattr(emb_outputs[0].outputs, "embedding"):
                embeddings = [out.outputs.embedding for out in emb_outputs]
            else:
                raise ValueError("Cannot extract embedding from model output")
            
            self.cache[query] = torch.tensor(embeddings[0])
            ret.append(torch.tensor(embeddings[0]))
            
            #if len(self.cache) % 10 == 0:  
            #    self._save_cache()
        result = torch.stack(ret)    
        return result
    
    def save_cache_final(self):
        self._save_cache()