from rapidfuzz import process, fuzz
import pandas as pd
import json
import tqdm
import sys
import pickle
import numpy as np
import faiss

import sys
import os
#sys.path.insert(-1, os.getcwd())

from codegraph_parser.python.codegraph_python_local import parse, NodeType, EdgeType
from datasets import load_dataset, Dataset
import os
import subprocess
import re

def extract_info(item):
    
    return item[1]


def get_extractor_anchor(graph, entity_query, keywords_query):

    cand_name_list = []
    cand_path_name_list = []

    for node in graph:
        node_type = node["type"]
        
        cand_name_list.append((node["id"], node["name"]))
        
        if node_type in [1, 2] :
            name_with_path = node["path"]
            cand_path_name_list.append((node["id"], name_with_path))

    cand_name_all = []
    cand_path_name_all = []

    for query in entity_query + keywords_query:
        
        if "/" in query:
            cand_path_name = process.extract((-1, query), cand_path_name_list, scorer=fuzz.WRatio, limit=3, processor=extract_info)
            cand_path_name_all.append(cand_path_name)

        query_wo_path = query.split('/')[-1]
        cand_name = process.extract((-1, query_wo_path), cand_name_list, scorer=fuzz.WRatio, limit=3, processor=extract_info)
        cand_name_all.append(cand_name)
            

    res = set()
    for query in cand_name_all:
        for item in query:
            res.add(item[0][0])
    for query in cand_path_name_all:
        for item in query:
            res.add(item[0][0])

    return res


def get_inferer_anchor(query_emb, node_embedding, k=15):

    query_emb = query_emb.numpy()
    raw_node_embedding = node_embedding.numpy()


    
    d = raw_node_embedding.shape[-1]#1024
    nb = len(raw_node_embedding)
    nq = 5

    index = faiss.IndexFlatL2(d)
    index.add(raw_node_embedding)
    D, I = index.search(raw_node_embedding[:5], k)
    D, I = index.search(query_emb, k)

    anchor_node = []
    for query in I:
        tmp_node_list = []
        for trans_id in query:
            tmp_node_list.append(trans_id)
        anchor_node.append(tmp_node_list)

    return anchor_node

    


def extract_modified_functions_and_classes_from_patch(patch_content):
    
    if not patch_content or patch_content.strip() == "":
        return [], []
    
    modified_functions = []
    modified_classes = []
    
    
    lines = patch_content.split('\n')
    
    
    current_hunk_functions = set()
    current_hunk_classes = set()
    has_modifications_in_hunk = False
    
    for line in lines:
        original_line = line
        stripped_line = line.strip()
        
        
        if stripped_line.startswith('@@') and '@@' in stripped_line[2:]:
            
            if has_modifications_in_hunk:
                modified_functions.extend([f for f in current_hunk_functions if f not in modified_functions])
                modified_classes.extend([c for c in current_hunk_classes if c not in modified_classes])
            
            
            current_hunk_functions.clear()
            current_hunk_classes.clear()
            has_modifications_in_hunk = False
            
            
            parts = stripped_line.split('@@')
            if len(parts) >= 3:
                context_line = parts[2].strip()
                
                
                class_match = re.match(r'class\s+(\w+)(?:\s*\([^)]*\))?\s*:', context_line)
                if class_match:
                    class_name = class_match.group(1)
                    current_hunk_classes.add(class_name)
                
                
                func_match = re.match(r'def\s+(\w+)\s*\(', context_line)
                if func_match:
                    func_name = func_match.group(1)
                    current_hunk_functions.add(func_name)
        
        
        elif stripped_line.startswith('+++') or stripped_line.startswith('---'):
            continue
            
        
        elif stripped_line.startswith(('+', '-')):
            has_modifications_in_hunk = True
            content = stripped_line[1:].strip()
            
            
            class_match = re.match(r'class\s+(\w+)(?:\s*\([^)]*\))?\s*:', content)
            if class_match:
                class_name = class_match.group(1)
                current_hunk_classes.add(class_name)
            
            
            func_match = re.match(r'def\s+(\w+)\s*\(', content)
            if func_match:
                func_name = func_match.group(1)
                current_hunk_functions.add(func_name)
        
        
        else:
            content = stripped_line
            
            
            class_match = re.match(r'class\s+(\w+)(?:\s*\([^)]*\))?\s*:', content)
            if class_match:
                class_name = class_match.group(1)
                current_hunk_classes.add(class_name)
            
            
            func_match = re.match(r'def\s+(\w+)\s*\(', content)
            if func_match:
                func_name = func_match.group(1)
                current_hunk_functions.add(func_name)
    
    
    if has_modifications_in_hunk:
        modified_functions.extend([f for f in current_hunk_functions if f not in modified_functions])
        modified_classes.extend([c for c in current_hunk_classes if c not in modified_classes])
    
    return modified_functions, modified_classes

def get_patch_related_node_ids(file_patch_mapping, subgraphdataset):
    
    patch_node_ids = []
    
    for file_path, patch_content in file_patch_mapping.items():
        if not patch_content or patch_content.strip() == "":
            continue
            
        
        modified_functions, modified_classes = extract_modified_functions_and_classes_from_patch(patch_content)
        
        
        if not file_path.startswith('/'):
            file_path = '/' + file_path
        
        print(f"Processing file: {file_path}")
        print(f"Modified functions: {modified_functions}")
        print(f"Modified classes: {modified_classes}")
        
        
        file_nodes = subgraphdataset.filter(lambda x: x["path"] == file_path)
        
        
        for class_name in modified_classes:
            matching_items = []
            
            for node in file_nodes:
                if node["type"] != 3:
                    continue
                    
                node_name = node["name"]
                node_id = node["id"]
                
                
                matched = False
                
                
                if node_name == class_name or node_name == f".{class_name}":
                    matched = True
                    
                
                elif node_name.endswith(f".{class_name}"):
                    matched = True
                
                
                elif ("attr" in node and 
                      isinstance(node["attr"], dict) and 
                      "code" in node["attr"]):
                    code = node["attr"]["code"]
                    if f"class {class_name}" in code or f"class {class_name}(" in code:
                        matched = True
                
                if matched and node_id not in [item["id"] for item in matching_items]:
                    matching_items.append(node)
            
            print(f"Found {len(matching_items)} matches for class '{class_name}': {[item['name'] for item in matching_items]}")
            patch_node_ids.extend([item["id"] for item in matching_items])
        
        # (type=4)  
        for func_name in modified_functions:
            matching_items = []
            
            for node in file_nodes:
                if node["type"] != 4:
                    continue
                    
                node_name = node["name"]
                node_id = node["id"]
                
                
                matched = False
                
                
                if node_name == func_name or node_name == f".{func_name}":
                    matched = True
                
                
                elif '.' in node_name:
                    name_parts = node_name.split('.')
                   
                    if name_parts[-1] == func_name:
                        matched = True
                    
                
                elif any(node_name == f".{class_name}.{func_name}" for class_name in modified_classes):
                    matched = True
                
                
                elif ("attr" in node and 
                      isinstance(node["attr"], dict) and 
                      "code" in node["attr"]):
                    code = node["attr"]["code"]
                    
                    if (re.search(rf'\bdef\s+{re.escape(func_name)}\s*\(', code) or
                        re.search(rf'\bdef\s+{re.escape(func_name)}\s*:', code)):
                        matched = True
                
                if matched and node_id not in [item["id"] for item in matching_items]:
                    matching_items.append(node)
            
            print(f"Found {len(matching_items)} matches for function '{func_name}': {[item['name'] for item in matching_items]}")
            patch_node_ids.extend([item["id"] for item in matching_items])
    
    
    patch_node_ids = list(set(patch_node_ids))
    print(f"Total patch-related node IDs: {len(patch_node_ids)}")
    
    return patch_node_ids

def get_graph_file_name(item):
    """
    return graph_file_name
    """

    raise NotImplementedError



if __name__ == "__main__":
    from datasets import Dataset
    import torch
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="astropy")
    args = parser.parse_args()
    with open(f"Graph_Feature_Construction/{args.repo}/rewriter_output_post.json", "r", encoding="utf-8") as file:
        rewriter_output = json.load(file)

    reponame = args.repo
    pulldataset = Dataset.load_from_disk(f"pulldata/{reponame}_postpro/")
    textgraphdataset = Dataset.load_from_disk(f"savedata/repos/{reponame}/")
    edgegraph = torch.load(f"pyggraph/{reponame}.timed.pt", weights_only=False)
    
    from safetensors import safe_open
    if os.path.exists(f"Graph_Feature_Construction/get_content_embedding/output_{reponame}/{reponame}_embeddings.safetensors"):
        with safe_open(f"/data/wangjuntong/GREPO/Graph_Feature_Construction/get_content_embedding/output_{reponame}/{reponame}_embeddings.safetensors", framework="pt", device='cpu') as f:
            emb_slice = f.get_slice("embeddings")
    else:
        with safe_open(f"Graph_Feature_Construction/get_content_embedding/{reponame}/{reponame}_embeddings.safetensors", framework="pt", device='cpu') as f:
            emb_slice = f.get_slice("embeddings")

    from safetensors import safe_open
    query_emb_slice = safe_open(f"Graph_Feature_Construction/query_embeddings_output_{reponame}/issue_query_embeddings.safetensors", framework="pt", device='cpu')# as f:
    #    query_emb_slice = f#.get_slice("embeddings")


    starttime, endtime = edgegraph.starttimestamp, edgegraph.endtimestamp
    from commit_utils import CommitDAGAnalyzer
    dag = CommitDAGAnalyzer(reponame)
    
    
    sha_path = []
    tmp = set()
    for commit_id in textgraphdataset["start_commit"]:
        if commit_id not in tmp:
            tmp.add(commit_id)
            sha_path.append(commit_id)
    
    
    commit_to_idx_mapping = {commit: i for i, commit in enumerate(sha_path)}


    # save path
    anchor_node_dict = {}

    for idx in rewriter_output.keys():
        pull_request = pulldataset[int(idx)] #['number', 'title', 'state', 'body', 'base_commit_sha', 'timestamp', 'files', 'file_patches', 'commit', 'review_comment', 'comment', 'review', 'issues', 'issues_info', 'participants', 'raw']
        
        base_commit_sha = pull_request["base_commit_sha"]
        if base_commit_sha in commit_to_idx_mapping:
            issue_time = commit_to_idx_mapping[base_commit_sha]
        else:
            print(f"Warning: base_commit_sha {base_commit_sha} not found in sha_path, skipping...")
            continue
            
        graphindice = torch.nonzero(torch.logical_and(issue_time>=starttime, issue_time<endtime)).flatten()
        
        subgraphdataset = textgraphdataset.select(graphindice.tolist())
        
        tarfiles = pull_request["files"]
        tarpatch = pull_request["file_patches"]
        
        
        #diff_files = parse_diff_files(tarpatch)
        #‘#@！@#’
        
        tarpatch_list = tarpatch.split("#@!@#")
        tarfiles_list = tarfiles.split(';')
        
        
        file_patch_mapping = {}
        for i, file_path in enumerate(tarfiles_list):
            if i < len(tarpatch_list):
                file_patch_mapping[file_path] = tarpatch_list[i]
        

        
        entity_query = rewriter_output[idx]["code_entity"]
        keyword_query = rewriter_output[idx]["keyword"]

        
        res_extractor = get_extractor_anchor(subgraphdataset, entity_query, keyword_query)
        
        res_inferer = get_inferer_anchor(query_emb_slice.get_tensor("issue_"+idx), emb_slice[graphindice])

        if tarfiles:
            file_paths = tarfiles.split(';')
            processed_paths = []
            for path in file_paths:
                if not path.startswith('/'):
                    processed_paths.append('/' + path)
                else:
                    processed_paths.append(path)
            tarfiles = ';'.join(processed_paths)
        tarfiles_ids = []
        target_path = tarfiles.split(';')
        for path in target_path:
            matching_items = subgraphdataset.filter(lambda x: x["path"] == path and x["type"] in [1,2])
            tarfiles_ids.extend([item["id"] for item in matching_items])
        #res_inferer = get_inferer_anchor(query_emb, tmp_node_embedding)
        
        
        patch_related_node_ids = get_patch_related_node_ids(file_patch_mapping, subgraphdataset)
        
        target_path = tarfiles.split(';')
        for path in target_path:
            matching_items = subgraphdataset.filter(lambda x: x["path"] == path and x["type"] in [1,2])
            if len(matching_items) > 0:
                target_commits = [(item["id"], item["start_commit"], item["end_commit"]) for item in matching_items]
                print(f"Found {len(target_commits)} items for path {path}")
                for item_id, start_commit, end_commit in target_commits:
                    print(f"ID: {item_id}, Start: {start_commit}, End: {end_commit}")
                    
                    start_time = commit_to_idx_mapping.get(start_commit, -1)
                    end_time = commit_to_idx_mapping.get(end_commit, -1)
                    print(f"start_time:{start_time}, end_time:{end_time}")
            else:
                print(f"No items found for path {path}")

        anchor_node = {
            "issue_time": int(issue_time),
            "extractor_anchor_nodes": [int(x) for x in res_extractor],
            "inferer_anchor_nodes": [[int(y) for y in x] for x in res_inferer],
            "tarfiles_ids": [int(x) for x in tarfiles_ids],
            "patch_related_node_ids": [int(x) for x in patch_related_node_ids]
        }
        
        anchor_node_dict[idx] = anchor_node
        
        # save
    with open(f"Graph_Feature_Construction/{args.repo}/anchor_node.json", 'w', encoding='utf-8') as file:
        json.dump(anchor_node_dict, file)
