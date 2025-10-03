from sentence_transformers import SentenceTransformer
import datasets
from torch_geometric.data import Data as PygData
from torch_geometric.nn.aggr import MinAggregation
import torch
import networkx as nx
import matplotlib.pyplot as plt
import sys
import subprocess
import git
import os
import argparse
import gc  
from memory_optimized_graph import create_graph_with_memory_optimization
from modelscope import AutoTokenizer, AutoModel
from typing import Dict, List, Any
from torch import Tensor
import os
os.environ['MLXLM_USE_MODELSCOPE'] = 'True'
from mlx_lm import load, generate

def commitid2time(repo_path):
    try:
        subprocess.check_output(f"cd {repo_path}; git checkout main", shell=True)
    except:
        subprocess.check_output(f"cd {repo_path}; git checkout -f master", shell=True)
    repo = git.Repo(repo_path)
    ret = {"none": sys.maxsize}
    for commit in repo.iter_commits():
        ret[str(commit)] = commit.committed_date
        ret[repr(str(commit))] = commit.committed_date
    return ret

def buildedge(data):
    ei = []  
    ea = []  
    relations = [
        ("contain", 0, 1),
        ("previous", 6, 7),
        ("superclasses", 4, 5),
        ("call", 2, 3)
    ]
    
    for rel, attr_fwd, attr_rev in relations:
    
        tar_list = data[rel]
        
        lencn = torch.tensor([len(t) for t in tar_list], dtype=torch.long)
        
        src = torch.arange(len(tar_list), dtype=torch.long).repeat_interleave(lencn)
        
        tar = torch.cat([torch.tensor(t, dtype=torch.long) for t in tar_list])
        
       
        ei.append(torch.stack((src, tar)))
        ea.append(torch.full((len(src),), attr_fwd, dtype=torch.long))
        
        
        ei.append(torch.stack((tar, src)))
        ea.append(torch.full((len(src),), attr_rev, dtype=torch.long))
    

    ei = torch.cat(ei, dim=1)
    ea = torch.cat(ea, dim=0)
    
    mask = (ei[0] >= 0) & (ei[1] >= 0)
    return ei[:, mask], ea[mask]

def proptime(data):
    f2cei = data.edge_index[:, data.edge_attr==0]
    nendtime = data.endtimestamp.scatter_reduce(
        0, 
        f2cei[1], 
        data.endtimestamp[f2cei[0]], 
        reduce="min", 
        include_self=True
    )
    
    while not torch.all(nendtime == data.endtimestamp):
        data.endtimestamp = nendtime
        nendtime = data.endtimestamp.scatter_reduce(
            0, 
            f2cei[1], 
            data.endtimestamp[f2cei[0]], 
            reduce="min", 
            include_self=True
        )
    return data

def callclosure(data):
    N = data.endtimestamp.shape[0]
    while True:
        caller2callee_ei = data.edge_index[:, data.edge_attr==2]
        previous2next_ei = data.edge_index[:, data.edge_attr==7]        
        adj1 = torch.sparse_coo_tensor(
            indices=caller2callee_ei, 
            values=torch.ones_like(caller2callee_ei[0], dtype=torch.float), 
            size=(N, N)
        )
        adj2 = torch.sparse_coo_tensor(
            indices=previous2next_ei, 
            values=torch.ones_like(previous2next_ei[0], dtype=torch.float), 
            size=(N, N)
        )

        adj = (adj1 @ adj2).coalesce()
        adj = torch.sparse_coo_tensor(
            indices=adj.indices(), 
            values=torch.ones_like(adj.values(), dtype=torch.float), 
            size=(N, N)
        ) - adj1
        adj = adj.coalesce()
        newcaller2callee = adj.indices()[:, adj.values() > 0.5]
        availmask = data.starttimestamp[newcaller2callee[0]] < data.endtimestamp[newcaller2callee[1]]
        availmask = torch.logical_and(
            availmask, 
            data.starttimestamp[newcaller2callee[1]] < data.endtimestamp[newcaller2callee[0]]
        )
        if not torch.any(availmask):
            break
        newcaller2callee = newcaller2callee[:, availmask]

        print(f"Added {newcaller2callee.shape[-1]} new edges", flush=True)

        newei = torch.concat((newcaller2callee, newcaller2callee[[1, 0]]), dim=-1)
        newea = torch.concat(
            (torch.zeros_like(newcaller2callee[0])+2, 
             torch.zeros_like(newcaller2callee[0])+3), 
            dim=-1
        )

        data.edge_index = torch.concat((data.edge_index, newei), dim=-1)
        data.edge_attr = torch.concat((data.edge_attr, newea), dim=-1)

    return data

# tokenizer = AutoTokenizer.from_pretrained(
#          'Qwen/Qwen3-Embedding-8B',
#          padding_side='left',
#          trust_remote_code=True,  
#          local_files_only=os.path.exists('Qwen/Qwen3-Embedding-8B')
#      )
def build_graph(name):

    data = datasets.Dataset.load_from_disk(f"savedata/repos/{name}/")
    print("Building edges...", flush=True)
    ei, ea = buildedge(data)
    
    
    print("Building commit paths...", flush=True)
    from commit_utils import CommitDAGAnalyzer
    analyzer = CommitDAGAnalyzer(repo_name=name)
    sha_path = analyzer.get_longest_path()
    commitid2timestamp = {sha: i for i, sha in enumerate(sha_path)}
    commitid2timestamp['none'] = sys.maxsize

    
    print("Processing timestamps...", flush=True)
    def batch_timetransform(batch):
        print("Processing batch timestamps...", flush=True)
        start_timestamps = []
        end_timestamps = []
        batch_size = len(batch["start_commit"])
        for start, end in zip(batch["start_commit"], batch["end_commit"]):
            start_ts = commitid2timestamp.get(start, commitid2timestamp['none'])
            end_ts = commitid2timestamp.get(end, commitid2timestamp['none'])
            start_timestamps.append(start_ts)
            end_timestamps.append(end_ts)
        return {
            "starttimestamp": start_timestamps,
            "endtimestamp": end_timestamps,
            "name_emb": [0] * batch_size,
            "attr_emb": [0] * batch_size
        }
    
    timedata = data.map(
        batch_timetransform,
        batched=True,
        batch_size=8192,
        num_proc=48
    )

    
    print("Generating text embeddings...", flush=True)
    #model = SentenceTransformer('all-mpnet-base-v2', local_files_only=False, device="cuda:0")#sentence-transformers/
    #model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B', local_files_only=True if os.path.exists('Qwen/Qwen3-Embedding-8B') else False).eval().to("cuda")
    #tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left', local_files_only=True if os.path.exists('Qwen/Qwen3-Embedding-8B') else False)
    #model, tokenizer = load("mlx-community/Qwen3-Embedding-8B-4bit-DWQ")
    #max_length = 512
    import torch
    from transformers import AutoModel, AutoTokenizer
    import os
    from transformers import BitsAndBytesConfig
    import importlib.metadata
    from packaging import version

    
    try:
        
        import bitsandbytes
        
        bnb_version = importlib.metadata.version("bitsandbytes")
        if version.parse(bnb_version) < version.parse("0.41.1"):
            
            print("pls run: pip install -U bitsandbytes")
    except ImportError:
        
        print("pls run: pip install bitsandbytes")
       
        use_quantization = False
    else:
        use_quantization = True

    
    if use_quantization:
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        
        quantization_config = None

    
    # model = AutoModel.from_pretrained(
    #     'Qwen/Qwen3-Embedding-8B',
    #     quantization_config=quantization_config,
    #     device_map="cuda:0",
    #     trust_remote_code=True,
    #     local_files_only=os.path.exists('Qwen/Qwen3-Embedding-8B')
    # ).eval()


    # max_length = 512

    # def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    #     return last_hidden_states[:, -1]
    #     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    #     if left_padding:
    #         return last_hidden_states[:, -1]
    #     else:
    #         sequence_lengths = attention_mask.sum(dim=1) - 1
    #         batch_size = last_hidden_states.shape[0]
    #         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    # def get_embeddings(input_texts: list, batch_size: int = 128, max_length: int = 512) -> torch.Tensor:
    #     all_embeddings = []
        
    #     batch_dict = tokenizer(
    #         input_texts,
    #         padding=True,
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt",
    #     )
    #     batch_dict = {k: v.to(model.device, non_blocking=True) for k, v in batch_dict.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch_dict)
    #         embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[:, :768]
    #         #all_embeddings.append(embeddings.cpu())
    #     return embeddings.cpu() #torch.cat(all_embeddings, dim=0)
    # from vllm import LLM
    # # model = LLM(
    # #         model="/home/wangjuntong/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B",
    # #         task="embed",
    # #         enforce_eager=True,

    # #     )
    # #tokenizer = model.get_tokenizer()
    # def embtransform(indict):
    #     bslen = len(indict["name"])
    #     assert len(indict["attr"]) == bslen
    #     pn = [a + ":" + b for a, b in zip(indict["path"], indict["name"])]
        
    #     msgs = indict["attr"]
    #     #emb = get_embeddings(msgs, batch_size=256, max_length=max_length)

    #     #prompt_token_id = tokenizer(msgs,return_tensors = 'pt')[:,:32768]
    #     max_length = 4096
    #     prompt_token_id_pn = tokenizer(
    #         pn,
    #         padding=True,
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt",
    #     )

    #     prompt_pn = tokenizer.batch_decode(prompt_token_id_pn["input_ids"], skip_special_tokens=True)
    #     #print(prompt)

    #     emb_outputs_pn = model.embed(prompt_pn)

    #     if hasattr(emb_outputs_pn[0], "embedding"):
    #         emb_pn = [out.embedding for out in emb_outputs_pn]
    #     elif hasattr(emb_outputs_pn[0], "hidden_states"):
    #         emb_pn = [out.hidden_states for out in emb_outputs_pn]
    #     elif hasattr(emb_outputs_pn[0], "outputs") and hasattr(emb_outputs_pn[0].outputs, "embedding"):
    #         emb_pn = [out.outputs.embedding for out in emb_outputs_pn]
    #     elif hasattr(emb_outputs_pn[0], "outputs") and hasattr(emb_outputs_pn[0].outputs, "hidden_states"):
    #         emb_pn = [out.outputs.hidden_states for out in emb_outputs_pn]
    #     else:
    #         raise ValueError("Cannot extract embedding from model.embed output; please check the output structure.")

    #     # Convert to list for pyarrow compatibility
    #     emb_pn = [e.tolist() if hasattr(e, "tolist") else list(e) for e in emb_pn]
        
    #     prompt_token_id_msgs = tokenizer(
    #         msgs,
    #         padding=True,
    #         truncation=True,
    #         max_length=max_length,
    #         return_tensors="pt",
    #     )

    #     prompt_msgs = tokenizer.batch_decode(prompt_token_id_msgs["input_ids"], skip_special_tokens=True)
    #     #print(prompt)

    #     emb_outputs_msgs = model.embed(prompt_msgs)

    #     if hasattr(emb_outputs_msgs[0], "embedding"):
    #         emb_msgs = [out.embedding for out in emb_outputs_msgs]
    #     elif hasattr(emb_outputs_msgs[0], "hidden_states"):
    #         emb_msgs = [out.hidden_states for out in emb_outputs_msgs]
    #     elif hasattr(emb_outputs_msgs[0], "outputs") and hasattr(emb_outputs_msgs[0].outputs, "embedding"):
    #         emb_msgs = [out.outputs.embedding for out in emb_outputs_msgs]
    #     elif hasattr(emb_outputs_pn[0], "outputs") and hasattr(emb_outputs_pn[0].outputs, "hidden_states"):
    #         emb_pn = [out.outputs.hidden_states for out in emb_outputs_pn]
    #     else:
    #         raise ValueError("Cannot extract embedding from model.embed output; please check the output structure.")

    #     # Convert to list for pyarrow compatibility
    #     emb_msgs = [e.tolist() if hasattr(e, "tolist") else list(e) for e in emb_msgs]

        
    #     return {
    #         "name_emb": emb_pn,
    #         "attr_emb": emb_msgs
    #     }


    # embdata = timedata.map(
    #     embtransform,
    #     batched=True,
    #     batch_size=160000  # Reduced batch size for embedding processing
    # )
    embdata = timedata
    
    print("Constructing graph...", flush=True)
    
    try:
        
        graph = create_graph_with_memory_optimization(data, embdata, timedata, ei, ea, name)
        
        os.makedirs("pyggraph", exist_ok=True)
        intermediate_path = f"pyggraph/{name}.pt"
        print(f"Saving graph to {intermediate_path}...", flush=True)
        torch.save(graph, intermediate_path)
        print(f"Saved intermediate graph to {intermediate_path}", flush=True)
        return intermediate_path
        
    except Exception as e:
        print(f"Memory optimized graph creation failed: {e}", flush=True)
        print("Attempting fallback with even smaller batches...", flush=True)
        
        
        try:
            graph = create_graph_with_memory_optimization(data, embdata, timedata, ei, ea, name, batch_size=500)
            
            os.makedirs("pyggraph", exist_ok=True)
            intermediate_path = f"pyggraph/{name}.pt"
            print(f"Saving graph to {intermediate_path}...", flush=True)
            torch.save(graph, intermediate_path)
            print(f"Saved intermediate graph to {intermediate_path}", flush=True)
            return intermediate_path
            
        except Exception as e2:
            print(f"All graph creation attempts failed: {e2}", flush=True)
            raise RuntimeError(f"Cannot create graph due to memory constraints: {e2}")

def process_graph(name):
    
    input_path = f"pyggraph/{name}.pt"
    output_path = f"pyggraph/{name}.timed.pt"
    
    print(f"Loading intermediate graph from {input_path}...", flush=True)
    data = torch.load(input_path, weights_only=False)
    data.edge_index = data.edge_index.to(torch.long)
    data.edge_attr = data.edge_attr.to(torch.long)
    
    print("Propagating timestamps...", flush=True)
    data = proptime(data)
    
    print("Computing call closure...", flush=True)
    data = callclosure(data)
    
    torch.save(data, output_path)
    print(f"Saved processed graph to {output_path}", flush=True)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Process repository graph data")
    parser.add_argument("name", type=str, help="Repository name")
    
    args = parser.parse_args()
    
    
    build_graph(args.name)
    
   
    process_graph(args.name)

if __name__ == "__main__":
    main()