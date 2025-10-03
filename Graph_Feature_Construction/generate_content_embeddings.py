import torch
import datasets
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm
import gc

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
    print("safetensors available")
except ImportError:
    print("safetensors not available, to save in numpy format")
    SAFETENSORS_AVAILABLE = False


try:
    from vllm import LLM
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
    print("VLLM AVAILABLE")
except ImportError:
    print("VLLM NOT AVAILABLE")
    VLLM_AVAILABLE = False

class ContentEmbeddingGenerator:
    
    
    def __init__(self, 
                 repo_name: str = "astropy",
                 model_path: str = "Qwen3-Embedding-8B",
                 output_dir: str = "./output"):
  
        self.repo_name = repo_name
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.graph_file = f"pyggraph/{repo_name}.timed.pt"
        self.dataset_dir = f"savedata/repos/{repo_name}/"
        
        
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        
        if not VLLM_AVAILABLE:
            print("VLLM NOT AVAILABLE")
            return
            
        try:
            print(f"Loading embedding model: {self.model_path}")
            self.model = LLM(
                model=self.model_path,
                task="embed",
                enforce_eager=True,
                gpu_memory_utilization=0.9,
                max_model_len=32768
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                padding_side='left',
                trust_remote_code=True
            )
            print("Loading embedding model finished")
            
        except Exception as e:
            print(f"Failured: {e}")
            self.model = None
            self.tokenizer = None
    
    def load_graph_data(self) -> torch.Tensor:
        
        print(f"Loading graph: {self.graph_file}")
        
        try:
            data = torch.load(self.graph_file, map_location='cpu', weights_only=False)
            num_nodes = data.num_nodes
            print(f"Loading graph finished with {num_nodes} nodes")
            
            
            node_ids = torch.arange(num_nodes, dtype=torch.long)
            return node_ids
            
        except Exception as e:
            print(f"Loading graph failured: {e}")
            raise
    
    def load_dataset(self) -> datasets.Dataset:
        
        print(f"Loading Datasets: {self.dataset_dir}")
        
        try:
            dataset = datasets.Dataset.load_from_disk(self.dataset_dir)
            print(f"Loading Datasets Finished with {len(dataset)} instances")
            return dataset
            
        except Exception as e:
            print(f"Loading Datasets failured: {e}")
            raise
    
    def extract_node_contents(self, node_ids: torch.Tensor, dataset: datasets.Dataset) -> Dict[int, Dict]:
        
        print("Extract node content...")
        
        node_contents = {}
        dataset_len = len(dataset)
        
        for node_id in tqdm(node_ids, desc="Extract node content"):
            node_id_int = int(node_id)
            
            if node_id_int < dataset_len:
                
                sample = dataset[node_id_int]
                
                node_contents[node_id_int] = {
                    'node_id': node_id_int,
                    'name': sample.get('name', ''),
                    'path': sample.get('path', ''),
                    'attr': sample.get('attr', ''),  
                    'type': sample.get('type', ''),
                    'start_commit': sample.get('start_commit', ''),
                    'end_commit': sample.get('end_commit', '')
                }
            else:
                
                node_contents[node_id_int] = {
                    'node_id': node_id_int,
                    'name': '',
                    'path': '',
                    'attr': '',
                    'type': '',
                    'start_commit': '',
                    'end_commit': ''
                }
        
        print(f"Extract node content Finished, with {sum(1 for v in node_contents.values() if v['attr'])} valid nodes")
        return node_contents
    
    def generate_embeddings(self, node_contents: Dict[int, Dict], batch_size: int = 32) -> Dict[int, np.ndarray]:
        
        if not self.model:
            print("models not loaded, skip")
            return {}
        
        print("generating embeddings...")
        
        
        texts = []
        node_id_order = []
        
        for node_id in sorted(node_contents.keys()):
            content = node_contents[node_id]
            text = content['name'] + content['path'] + content['attr']

            if not text or text.strip() == '{}' or text.strip() == '':
                text = " "  
            
            texts.append(text)
            node_id_order.append(node_id)
        
        print(f"number of texts need to generate embedding: {len(texts)}")
        
        
        embeddings = {}
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(texts), batch_size), desc="generate embeddings", total=total_batches):
            batch_texts = texts[i:i + batch_size]
            batch_node_ids = node_id_order[i:i + batch_size]
            
            try:
                
                tokenized = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=32768,
                    return_tensors="pt"
                )
                
                
                processed_texts = self.tokenizer.batch_decode(
                    tokenized["input_ids"], 
                    skip_special_tokens=True
                )
                
                
                emb_outputs = self.model.embed(processed_texts)
                
                
                for j, node_id in enumerate(batch_node_ids):
                    emb_output = emb_outputs[j]
                    
                    
                    if hasattr(emb_output, "embedding"):
                        embedding = emb_output.embedding
                    elif hasattr(emb_output, "hidden_states"):
                        embedding = emb_output.hidden_states
                    elif hasattr(emb_output, "outputs") and hasattr(emb_output.outputs, "embedding"):
                        embedding = emb_output.outputs.embedding
                    elif hasattr(emb_output, "outputs") and hasattr(emb_output.outputs, "hidden_states"):
                        embedding = emb_output.outputs.hidden_states
                    else:
                        raise ValueError("Cannot extract embedding")
                    
                    
                    if hasattr(embedding, "cpu"):
                        embedding = embedding.cpu().numpy()
                    elif hasattr(embedding, "numpy"):
                        embedding = embedding.numpy()
                    else:
                        embedding = np.array(embedding)
                    
                    embeddings[node_id] = embedding
                
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Batch {i//batch_size + 1} embedding-generation FAILURED: {e}")
                
                for node_id in batch_node_ids:
                    embeddings[node_id] = np.zeros(4096, dtype=np.float32)
        
        print(f"Embedding Generation Finished: {len(embeddings)}")
        return embeddings
    
    def save_content_json(self, node_contents: Dict[int, Dict]):
        
        output_file = self.output_dir / f"{self.repo_name}_node_contents.json"
        print(f"Save node content to: {output_file}")
        
        
        sorted_contents = {}
        for node_id in sorted(node_contents.keys()):
            sorted_contents[str(node_id)] = node_contents[node_id]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sorted_contents, f, indent=2, ensure_ascii=False)
            print(f"Node content save finished")
            
        except Exception as e:
            print(f"Node content save failured: {e}")
            raise
    
    def save_embeddings_safetensor(self, embeddings: Dict[int, np.ndarray], all_node_ids: List[int] = None):
        
        if not embeddings:
            print("embedding not found, skip saving")
            return
        
        print("Saving embeddings...")
        
        
        if all_node_ids:
            for node_id in all_node_ids:
                if node_id not in embeddings:
                    print(f"Node-id {node_id} with NO embedding")
                    embeddings[node_id] = np.zeros(4096, dtype=np.float32)
        
        
        sorted_node_ids = sorted(embeddings.keys())
        embedding_list = []
        
        for node_id in sorted_node_ids:
            embedding = embeddings[node_id]
            if isinstance(embedding, np.ndarray):
                embedding_list.append(torch.from_numpy(embedding))
            else:
                embedding_list.append(torch.tensor(embedding))
        
        
        embeddings_tensor = torch.stack(embedding_list, dim=0)
        print(f"Embeddings tensor shape: {embeddings_tensor.shape}")
        
        
        print(f"NODE-ID: {min(sorted_node_ids)} - {max(sorted_node_ids)}")
        print(f"Whether node IDs consecutive: {sorted_node_ids == list(range(min(sorted_node_ids), max(sorted_node_ids) + 1))}")
        
        
        if SAFETENSORS_AVAILABLE:
            output_file = self.output_dir / f"{self.repo_name}_embeddings.safetensors"
            tensors = {"embeddings": embeddings_tensor}
            save_file(tensors, output_file)
            print(f"Safetensor saved to: {output_file}")
        else:
            output_file = self.output_dir / f"{self.repo_name}_embeddings.npz"
            np.savez_compressed(output_file, embeddings=embeddings_tensor.numpy())
            print(f"Numpy array saved to: {output_file}")
        
       
        index_file = self.output_dir / f"{self.repo_name}_embedding_index.json"
        index_mapping = {
            'node_ids': sorted_node_ids,
            'embedding_dim': embeddings_tensor.shape[1],
            'num_nodes': len(sorted_node_ids),
            'format': 'safetensors' if SAFETENSORS_AVAILABLE else 'numpy',
            'min_node_id': min(sorted_node_ids),
            'max_node_id': max(sorted_node_ids),
            'is_continuous': sorted_node_ids == list(range(min(sorted_node_ids), max(sorted_node_ids) + 1))
        }
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_mapping, f, indent=2)
        print(f"Index mapping saved to: {index_file}")
    
    def run(self, batch_size: int = 32):
        
        print(f"Start to process repo: {self.repo_name}")
        
        try:
            
            node_ids = self.load_graph_data()
            
            
            dataset = self.load_dataset()
            
            
            node_contents = self.extract_node_contents(node_ids, dataset)
            
            
            #self.save_content_json(node_contents)
            
            
            embeddings = self.generate_embeddings(node_contents, batch_size)
            
            
            self.save_embeddings_safetensor(embeddings, all_node_ids=[int(nid) for nid in node_ids])
            
            print(f"Finished！")
            print(f"Statistics:")
            print(f"number of nodes: {len(node_contents)}")
            print(f"nodes with content: {sum(1 for v in node_contents.values() if v['attr'] and v['attr'] != '{}')}")
            print(f"number of embeddings: {len(embeddings)}")
            
        except Exception as e:
            print(f"process failured: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    parser = argparse.ArgumentParser(description="generate node content and embedding")
    parser.add_argument('--repo_name', type=str, default='astropy', help='name of repo')
    parser.add_argument('--model_path', type=str, 
                       default='/data/wangjuntong/FROM_120/data1/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B',
                       help='Embedding-model path')
    parser.add_argument('--output_dir', type=str, default='./output', help='output_dir')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    
    args = parser.parse_args()
    
    generator = ContentEmbeddingGenerator(
        repo_name=args.repo_name,
        model_path=args.model_path,
        output_dir=args.output_dir
    )
    
    generator.run(batch_size=args.batch_size)

if __name__ == "__main__":
    main()
