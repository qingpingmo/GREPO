
import argparse
import json
import logging
import os
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import save_file, load_file

from vllm import LLM
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderedQueryEmbeddingGenerator:
    
    def __init__(self):
        self.model = LLM(
            model="Qwen3-Embedding-8B",
            task="embed",
            enforce_eager=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'Qwen3-Embedding-8B',
            padding_side='left',
            trust_remote_code=True,
            local_files_only=True
        )
        self.max_length = 32768
        logger.info("Query embedding generator initialization finished")
    
    def generate_single_query_embedding(self, query: str) -> torch.Tensor:

        if not query:
            return torch.zeros(1, 4096)
        
        
        prompt_token_ids = self.tokenizer(
            [query],  
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        prompt = self.tokenizer.batch_decode(
            prompt_token_ids["input_ids"], 
            skip_special_tokens=True
        )
        
        emb_outputs = self.model.embed(prompt)
        
        
        if hasattr(emb_outputs[0], "embedding"):
            embedding = emb_outputs[0].embedding
        elif hasattr(emb_outputs[0], "outputs") and hasattr(emb_outputs[0].outputs, "embedding"):
            embedding = emb_outputs[0].outputs.embedding
        else:
            raise ValueError("Cannot extract embedding from model output")
        
       
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)
        
        return embedding.unsqueeze(0)  
    
    def generate_batch_query_embeddings(self, queries: List[str], batch_size: int = 8) -> torch.Tensor:

        if not queries:
            return torch.zeros(0, 4096)
        
        logger.info(f"Start to generate {len(queries)} query's embeddings with batch-size: {batch_size}")
        
        all_embeddings = []
        
        
        for i in tqdm(range(0, len(queries), batch_size), desc="generate embeddings"):
            batch_queries = queries[i:i+batch_size]
            
            
            prompt_token_ids = self.tokenizer(
                batch_queries,
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
            
            
            batch_embeddings = []
            for out in emb_outputs:
                if hasattr(out, "embedding"):
                    emb = out.embedding
                elif hasattr(out, "outputs") and hasattr(out.outputs, "embedding"):
                    emb = out.outputs.embedding
                else:
                    raise ValueError("Cannot extract embedding from model output")
                
                if not isinstance(emb, torch.Tensor):
                    emb = torch.tensor(emb)
                batch_embeddings.append(emb)
            
            
            batch_tensor = torch.stack(batch_embeddings)
            all_embeddings.append(batch_tensor)
        
        
        result = torch.cat(all_embeddings, dim=0)
        logger.info(f"Generation finished, {result.shape[0]} embeddings, shape: {result.shape[1]}")
        
        return result


def load_rewriter_data(args):
    
    logger.info("Loading ...")
    
    with open(f'Graph_Feature_Construction/{args.repo}/rewriter_output_post.json', 'r') as f:
        rewriter_data = json.load(f)
    
    logger.info(f"Finished with {len(rewriter_data)} instances")
    return rewriter_data


def extract_queries_ordered(rewriter_data: Dict) -> Tuple[List[Tuple[str, List[str]]], List[str]]:

    logger.info("Extract queries in order...")
    
    issues_with_queries = []
    all_queries_flat = []
    
    
    sorted_issues = sorted(rewriter_data.items(), key=lambda x: int(x[0]))
    
    for issue_id, item in sorted_issues:
        queries = item.get('query', [])
        if queries:
            issues_with_queries.append((issue_id, queries))
            all_queries_flat.extend(queries)
    
    logger.info(f"Extraction Finished:")
    logger.info(f"  Valid issues: {len(issues_with_queries)}")
    logger.info(f"  All queries: {len(all_queries_flat)}")
    
    return issues_with_queries, all_queries_flat


def save_embeddings_ordered(embeddings: torch.Tensor, 
                           issues_with_queries: List[Tuple[str, List[str]]],
                           output_dir: str = "./query_embeddings_output"):

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Save embeddings to: {output_dir}")
    
    
    safetensor_path = os.path.join(output_dir, "all_query_embeddings.safetensors")
    save_file({"embeddings": embeddings}, safetensor_path)
    logger.info(f"Save in SafeTensors Format: {safetensor_path}")
    
    
    issue_embeddings_dict = {}
    embedding_idx = 0
    
    for issue_id, queries in issues_with_queries:
        num_queries = len(queries)
        issue_embeddings = embeddings[embedding_idx:embedding_idx + num_queries]
        
        
        issue_embeddings_dict[f"issue_{issue_id}"] = issue_embeddings
        
        embedding_idx += num_queries
    
    issue_safetensor_path = os.path.join(output_dir, "issue_query_embeddings.safetensors")
    save_file(issue_embeddings_dict, issue_safetensor_path)
    logger.info(f"Save SafeTensors by issues: {issue_safetensor_path}")
    
    
    # numpy_path = os.path.join(output_dir, "all_query_embeddings.npy")
    # np.save(numpy_path, embeddings.cpu().numpy())
    # 
    
    
    structured_data = {}
    embedding_idx = 0
    
    for issue_id, queries in issues_with_queries:
        num_queries = len(queries)
        
        structured_data[issue_id] = {
            'queries': queries,
            'num_queries': num_queries,
            'embedding_start_idx': embedding_idx,
            'embedding_end_idx': embedding_idx + num_queries,
            'embedding_shape': [num_queries, embeddings.shape[1]]
        }
        
        embedding_idx += num_queries
    
    
    # pickle_path = os.path.join(output_dir, "query_metadata.pkl")
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(structured_data, f)
    
    
    
    metadata = {
        'total_embeddings': embeddings.shape[0],
        'embedding_dim': embeddings.shape[1],
        'total_issues': len(issues_with_queries),
        'issues_order': [issue_id for issue_id, _ in issues_with_queries],
        'total_queries_per_issue': {issue_id: len(queries) for issue_id, queries in issues_with_queries},
        'file_formats': {
            'main_embeddings': 'all_query_embeddings.safetensors',
            'issue_embeddings': 'issue_query_embeddings.safetensors',
            'numpy_backup': 'all_query_embeddings.npy',
            'metadata': 'query_metadata.pkl'
        },
        'usage_examples': {
            'load_all_embeddings': "from safetensors.torch import load_file; embeddings = load_file('all_query_embeddings.safetensors')['embeddings']",
            'load_issue_embeddings': "from safetensors.torch import load_file; issue_embs = load_file('issue_query_embeddings.safetensors')",
            'load_metadata': "import pickle; with open('query_metadata.pkl', 'rb') as f: metadata = pickle.load(f)"
        }
    }
    
    # metadata_path = os.path.join(output_dir, "metadata.json")
    # with open(metadata_path, 'w') as f:
    #     json.dump(metadata, f, indent=2)
 
    return output_dir


def parse_args():
    
    parser = argparse.ArgumentParser(description='Generate and save query embeddings in order')
    
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='batch_size')
    parser.add_argument('--output_dir', type=str, default='./query_embeddings_output',
                       help='output_dir')
    parser.add_argument('--max_issues', type=int, default=-1,
                       help='max issues to be processed，-1 means all')
    parser.add_argument("--repo", type=str, default="astropy")
    return parser.parse_args()


def main():
    
    args = parse_args()
    
    logger.info("Start to generate query embeddings...")
    logger.info(f"parameters:")
    logger.info(f" batch_size: {args.batch_size}")
    logger.info(f" output_dir: {args.output_dir}")
    logger.info(f" max_issues: {args.max_issues if args.max_issues > 0 else 'all'}")
    
    
    rewriter_data = load_rewriter_data(args=args)
    
    
    issues_with_queries, all_queries_flat = extract_queries_ordered(rewriter_data)
    
    
    if args.max_issues > 0:
        issues_with_queries = issues_with_queries[:args.max_issues]
        
        all_queries_flat = []
        for _, queries in issues_with_queries:
            all_queries_flat.extend(queries)
        logger.info(f"Limited len: {len(issues_with_queries)} issues, {len(all_queries_flat)} queries")
    
    
    generator = OrderedQueryEmbeddingGenerator()
    embeddings = generator.generate_batch_query_embeddings(
        all_queries_flat, 
        batch_size=args.batch_size
    )
    
    
    output_dir = save_embeddings_ordered(embeddings, issues_with_queries, args.output_dir)
    



if __name__ == "__main__":
    main()
