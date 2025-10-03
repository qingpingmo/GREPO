import torch
import gc
import os
from torch_geometric.data import Data as PygData
import datasets

def create_graph_with_memory_optimization(data, embdata, timedata, ei, ea, name, batch_size=2000):

    print("Creating graph with memory optimization...", flush=True)
    
    total_samples = len(embdata)
    print(f"Total samples: {total_samples}", flush=True)
    
    
    try:
        
        test_batch = embdata.select(range(min(100, total_samples)))
        test_name_emb = torch.tensor(test_batch["name_emb"], dtype=torch.float)
        test_attr_emb = torch.tensor(test_batch["attr_emb"], dtype=torch.float)
        
       
        name_emb_size = test_name_emb.element_size() * test_name_emb.numel() * (total_samples / 100)
        attr_emb_size = test_attr_emb.element_size() * test_attr_emb.numel() * (total_samples / 100)
        total_emb_size = name_emb_size + attr_emb_size
        
        print(f"Estimated embedding memory: {total_emb_size / 1024**3:.2f} GB", flush=True)
        
        del test_batch, test_name_emb, test_attr_emb
        gc.collect()
        
        
        if total_emb_size > 16 * 1024**3:  # 16GB
            batch_size = max(1000, batch_size // 2)
            print(f"Large dataset detected, using smaller batch size: {batch_size}", flush=True)
        
    except Exception as e:
        print(f"Memory estimation failed: {e}, using conservative batch size", flush=True)
        batch_size = 1000
    
    
    print("Processing name embeddings in batches...", flush=True)
    name_emb_list = []
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_num = i // batch_size + 1
        total_batches = (total_samples + batch_size - 1) // batch_size
        print(f"Processing name embeddings batch {batch_num}/{total_batches} ({end_idx-i} samples)...", flush=True)
        
        try:
            batch_data = embdata.select(range(i, end_idx))
            name_batch = torch.tensor(batch_data["name_emb"], dtype=torch.float)
            name_emb_list.append(name_batch)
            del batch_data, name_batch
            gc.collect()
        except Exception as e:
            print(f"Error processing name embedding batch {batch_num}: {e}", flush=True)
            
            smaller_batch_size = batch_size // 2
            if smaller_batch_size < 100:
                raise RuntimeError("Cannot process data even with very small batches")
            
            for j in range(i, end_idx, smaller_batch_size):
                sub_end = min(j + smaller_batch_size, end_idx)
                batch_data = embdata.select(range(j, sub_end))
                name_batch = torch.tensor(batch_data["name_emb"], dtype=torch.float)
                name_emb_list.append(name_batch)
                del batch_data, name_batch
                gc.collect()
    
    print("Concatenating name embeddings...", flush=True)
    name_emb = torch.cat(name_emb_list, dim=0)
    del name_emb_list
    gc.collect()
    print(f"Name embeddings shape: {name_emb.shape}", flush=True)
    
    
    print("Processing attr embeddings in batches...", flush=True)
    attr_emb_list = []
    
    for i in range(0, total_samples, batch_size):
        end_idx = min(i + batch_size, total_samples)
        batch_num = i // batch_size + 1
        total_batches = (total_samples + batch_size - 1) // batch_size
        print(f"Processing attr embeddings batch {batch_num}/{total_batches} ({end_idx-i} samples)...", flush=True)
        
        try:
            batch_data = embdata.select(range(i, end_idx))
            attr_batch = torch.tensor(batch_data["attr_emb"], dtype=torch.float)
            attr_emb_list.append(attr_batch)
            del batch_data, attr_batch
            gc.collect()
        except Exception as e:
            print(f"Error processing attr embedding batch {batch_num}: {e}", flush=True)
            
            smaller_batch_size = batch_size // 2
            if smaller_batch_size < 100:
                raise RuntimeError("Cannot process data even with very small batches")
            
            for j in range(i, end_idx, smaller_batch_size):
                sub_end = min(j + smaller_batch_size, end_idx)
                batch_data = embdata.select(range(j, sub_end))
                attr_batch = torch.tensor(batch_data["attr_emb"], dtype=torch.float)
                attr_emb_list.append(attr_batch)
                del batch_data, attr_batch
                gc.collect()
    
    print("Concatenating attr embeddings...", flush=True)
    attr_emb = torch.cat(attr_emb_list, dim=0)
    del attr_emb_list
    gc.collect()
    print(f"Attr embeddings shape: {attr_emb.shape}", flush=True)
    
    
    print("Creating final graph data structure...", flush=True)
    try:
        graph = PygData(
            name=name_emb,
            type=torch.tensor(data["type"], dtype=torch.long),
            attr=attr_emb,
            edge_index=ei,
            edge_attr=ea,
            starttimestamp=torch.tensor(timedata["starttimestamp"], dtype=torch.long),
            endtimestamp=torch.tensor(timedata["endtimestamp"], dtype=torch.long)
        )
        graph.type = torch.tensor(graph.type, dtype=torch.int)
        
        
        del name_emb, attr_emb
        gc.collect()
        
        return graph
        
    except Exception as e:
        print(f"Error creating graph: {e}", flush=True)
        
        del name_emb, attr_emb
        gc.collect()
        raise


if __name__ == "__main__":
    
    pass
