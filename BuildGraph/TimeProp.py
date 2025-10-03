from sentence_transformers import SentenceTransformer
import datasets
from torch_geometric.data import Data as PygData
import torch
import networkx as nx
import matplotlib.pyplot as plt
import sys
import subprocess
from torch_geometric.nn.aggr import MinAggregation

def proptime(data):
    f2cei = data.edge_index[:, data.edge_attr==0]
    #nendtime: torch.Tensor = data.endtimestamp
    nendtime = data.endtimestamp.scatter_reduce(0, f2cei[1], data.endtimestamp[f2cei[0]], reduce="min", include_self=True)
    #MinAggregation().forward(data.endtimestamp[f2cei[0]], f2cei[1], dim=0, dim_size=data.endtimestamp.shape[0])
    while not torch.all(nendtime==data.endtimestamp):
        #print("step")
        data.endtimestamp = nendtime
        #nendtime = MinAggregation().forward(data.endtimestamp[f2cei[0]], f2cei[1], dim=0, dim_size=data.endtimestamp.shape[0])
        nendtime = data.endtimestamp.scatter_reduce(0, f2cei[1], data.endtimestamp[f2cei[0]], reduce="min", include_self=True)
    #print(nendtime)
    return data


def callclosure(data):
    N = data.endtimestamp.shape[0]
    while True:
        caller2callee_ei = data.edge_index[:, data.edge_attr==2]
        previous2next_ei = data.edge_index[:, data.edge_attr==7]        
        adj1 = torch.sparse_coo_tensor(indices=caller2callee_ei, values=torch.ones_like(caller2callee_ei[0], dtype=torch.float), size=(N, N))
        adj2 = torch.sparse_coo_tensor(indices=previous2next_ei, values=torch.ones_like(previous2next_ei[0], dtype=torch.float), size=(N, N))

        adj = (adj1 @ adj2).coalesce()
        adj = torch.sparse_coo_tensor(indices=adj.indices(), values=torch.ones_like(adj.values(), dtype=torch.float), size=(N, N)) - adj1
        adj = adj.coalesce()
        newcaller2callee = adj.indices()[:, adj.values() > 0.5]
        availmask = data.starttimestamp[newcaller2callee[0]]<data.endtimestamp[newcaller2callee[1]]
        availmask = torch.logical_and(availmask, data.starttimestamp[newcaller2callee[1]]<data.endtimestamp[newcaller2callee[0]])
        if not torch.any(availmask):
            break
        newcaller2callee = newcaller2callee[:, availmask]

        print("new edge", newcaller2callee.shape[-1], flush=True)

        newei = torch.concat((newcaller2callee, newcaller2callee[[1, 0]]), dim=-1)
        newea = torch.concat((torch.zeros_like(newcaller2callee[0])+2, torch.zeros_like(newcaller2callee[0])+3), dim=-1)

        data.edge_index = torch.concat((data.edge_index, newei), dim=-1)
        data.edge_attr = torch.concat((data.edge_attr, newea), dim=-1)


    #print(nendtime)
    return data


def main(name):
    data = torch.load(f"pyggraph/{name}.pt")
    data.edge_index = data.edge_index.to(torch.long)
    data.edge_attr = data.edge_attr.to(torch.long)
    data = proptime(data)
    data = callclosure(data)
    torch.save(data, f"pyggraph/{name}.timed.pt")

# Main function
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    args = parser.parse_args()
    main(args.name)