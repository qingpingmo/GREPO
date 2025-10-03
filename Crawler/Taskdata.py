import datasets
import subprocess
import sys
import pickle
from sentence_transformers import SentenceTransformer
from utils import logger
import os
from tasks.patch_utils import parse_patch
import numpy as np
from GithubRepoAnalysis import SPLIT_STR
from modelscope import AutoTokenizer, AutoModel

import json
def analyze_issue_info(issues_info_str_list: list[str], random_sent_gen) -> list[list[str]]:
    return_list = [[] for i in range(10)]
    for iis in issues_info_str_list:
        iiss = iis.split(SPLIT_STR)
        bug_desc = ""
        expected_behavior = ""
        actual_behavior = ""
        reproduce = ""
        require = ""
        code = ""
        error_trace = ""
        error_statement = ""
        title = ""
        others = ""
        for one_issue_info in iiss:
            # Skip empty strings to prevent JSON parsing errors
            if not one_issue_info.strip():
                continue  # Skip empty strings
            try:
                issue_data = json.loads(one_issue_info)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON
            title += issue_data.get("title", "")
            body_analysis = issue_data.get("body_analysis", {})  # Handle missing key
            
            bug_desc += body_analysis.get("bug_desc", "")
            expected_behavior += body_analysis.get("expected_behavior", "")
            actual_behavior += body_analysis.get("actual_behavior", "")
            reproduce += body_analysis.get("reproduce", "")
            require += body_analysis.get("require", "")
            code += body_analysis.get("code", "")
            error_trace += body_analysis.get("error_trace", "")
            error_statement += body_analysis.get("error_statement", "")
            others += body_analysis.get("others", "")
            print(f"Parsed issue info: {issue_data.get('title', '')}")
            print(f"Bug description: {bug_desc}")
            print(f"Expected behavior: {expected_behavior}")
            print(f"Actual behavior: {actual_behavior}")
            print(f"Reproduce steps: {reproduce}")
            print(f"Require: {require}")
            print(f"Code: {code}")
            print(f"Error trace: {error_trace}")
            print(f"Error statement: {error_statement}")
            print(f"Title: {title}")
            print(f"Others: {others}")
            print("--------------------------------------------------")
            
        return_list[0].append(empty_str_augment(bug_desc, 5, random_sent_gen))
        return_list[1].append(empty_str_augment(expected_behavior, 2, random_sent_gen))
        return_list[2].append(empty_str_augment(actual_behavior, 2, random_sent_gen))
        return_list[3].append(empty_str_augment(reproduce, 10, random_sent_gen))
        return_list[4].append(empty_str_augment(require, 5, random_sent_gen))
        return_list[5].append(empty_str_augment(code, 20, random_sent_gen))
        return_list[6].append(empty_str_augment(error_trace, 20, random_sent_gen))
        return_list[7].append(empty_str_augment(error_statement, 1, random_sent_gen))
        return_list[8].append(empty_str_augment(others, 1, random_sent_gen))
        return_list[9].append(title)
        
    return return_list

def empty_str_augment(str_: str, num_sent: int, rsg) -> str:
    if not str_ or str_.isspace():
        return rsg.generate_random_sentence(num_sent=num_sent)
    return str_


tokenizer = AutoTokenizer.from_pretrained(
         'Qwen/Qwen3-Embedding-8B',
         padding_side='left',
         trust_remote_code=True,  
         local_files_only=os.path.exists('Qwen/Qwen3-Embedding-8B')
     )

from vllm import LLM
model = LLM(
            model="/home/wangjuntong/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-8B",
            task="embed",
            enforce_eager=True,

        )

def main(REPO_NAME):
    from commit_utils import CommitDAGAnalyzer
    analyzer = CommitDAGAnalyzer(repo_name=REPO_NAME)
    sha_path = analyzer.get_longest_path()
    commit_id2time = {sha:i for i, sha in enumerate(sha_path)}

    name_version2id = datasets.Dataset.load_from_disk(f"savedata/repos/{REPO_NAME}")

    from GithubRepoAnalysis import SPLIT_STR

    pulldata = datasets.Dataset.load_from_disk(f"pulldata/{REPO_NAME}_postpro")
    pulldata = pulldata.filter(lambda sha: sha in commit_id2time, input_columns=["base_commit_sha"])
    from utils import RandomSentenceGenerator
    #model = SentenceTransformer('all-mpnet-base-v2', local_files_only=True, device="cuda:0")#sentence-transformers/
    #model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", local_files_only=False, device="cuda:0")
    from modelscope import AutoTokenizer, AutoModel
    from openai import OpenAI
    from torch import Tensor
    import json
    import torch
    def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_embeddings(input_texts, batch_size=4):
        model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B', local_files_only=True if os.path.exists('Qwen/Qwen3-Embedding-8B') else False).eval().to("cuda")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left', local_files_only=True if os.path.exists('Qwen/Qwen3-Embedding-8B') else False)
        max_length = 2048

        all_embeddings = []
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i+batch_size]
            batch_dict = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[:,:768]
                all_embeddings.append(embeddings.cpu())
        return torch.cat(all_embeddings, dim=0)
    
    def embtransform(indict):
        bslen = len(indict["title"])
        ret = {}
        issues_info = analyze_issue_info(indict["issues_info"], RandomSentenceGenerator())
        #emb = model.encode(issues_info[0] + issues_info[1] + issues_info[2] + issues_info[3] + issues_info[4] + issues_info[5] + issues_info[6] + issues_info[7] + issues_info[8])
        
        #emb = get_embeddings(issues_info[0] + issues_info[1] + issues_info[2] + issues_info[3] + issues_info[4] + issues_info[5] + issues_info[6] + issues_info[7] + issues_info[8])
        issues_msgs = issues_info[0] + issues_info[1] + issues_info[2] + issues_info[3] + issues_info[4] + issues_info[5] + issues_info[6] + issues_info[7] + issues_info[8]
        max_length = 16384
        prompt_token_id_issues_msgs = tokenizer(
            issues_msgs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        prompt_issues_msgs = tokenizer.batch_decode(prompt_token_id_issues_msgs["input_ids"], skip_special_tokens=True)
        emb_outputs_issues_msgs = model.embed(prompt_issues_msgs)
        if hasattr(emb_outputs_issues_msgs[0], "embedding"):
            emb_issues_msgs = [out.embedding for out in emb_outputs_issues_msgs]
        elif hasattr(emb_outputs_issues_msgs[0], "hidden_states"):
            emb_issues_msgs = [out.hidden_states for out in emb_outputs_issues_msgs]
        elif hasattr(emb_outputs_issues_msgs[0], "outputs") and hasattr(emb_outputs_issues_msgs[0].outputs, "embedding"):
            emb_issues_msgs = [out.outputs.embedding for out in emb_outputs_issues_msgs]
        elif hasattr(emb_outputs_issues_msgs[0], "outputs") and hasattr(emb_outputs_issues_msgs[0].outputs, "hidden_states"):
            emb_issues_msgs = [out.outputs.hidden_states for out in emb_outputs_issues_msgs]
        else:
            raise ValueError("Cannot extract embedding from model.embed output; please check the output structure.")

        emb = [e.tolist() if hasattr(e, "tolist") else list(e) for e in emb_issues_msgs]
        
        
        ret["bug_desc_emb"] = emb[:bslen]
        ret["expected_behavior_emb"] = emb[bslen:2*bslen]
        ret["actual_behavior_emb"] = emb[2*bslen:3*bslen]
        ret["reproduce_emb"] = emb[3*bslen:4*bslen]
        ret["require_emb"] = emb[4*bslen:5*bslen]
        ret["code_emb"] = emb[5*bslen:6*bslen]
        ret["error_trace_emb"] = emb[6*bslen:7*bslen]
        ret["error_statement_emb"] = emb[7*bslen:8*bslen]
        ret["issue_title_emb"] = emb[8*bslen:]
        return ret
       
    pulldata = pulldata.remove_columns("timestamp")   
    embpulldata = pulldata.map(embtransform, batched=True, batch_size=100000)
    print("len pulldata", len(embpulldata))
    #exit()
    # Step 1: Parse changed lines from the patch
    
    repo_path = f'repos/{REPO_NAME}'
    try:
        print(subprocess.check_output(f"cd {repo_path}; git checkout -f main", shell=True))
    except Exception:
        try:
            print(subprocess.check_output(f"cd {repo_path}; git checkout -f master", shell=True))
        except Exception:
            print(subprocess.check_output(f"cd {repo_path}; git checkout -f pre-commit-ci-update-config", shell=True))
            

    from RepoAnalysis import nodetypedict
    def committransform(indict):
        time = commit_id2time[indict["base_commit_sha"]]
        #print(f"Processing commit time {time} for repo {REPO_NAME}", flush=True)
        currgraph = name_version2id.filter(
            lambda start_commit, end_commit: 
                commit_id2time.get(start_commit, sys.maxsize) <= time < commit_id2time.get(end_commit, sys.maxsize),
            input_columns=["start_commit", "end_commit"]
        )
        
        patches = indict["file_patches"].split(SPLIT_STR)
        files = indict["files"].split(";")
        assert len(files) == len(patches)

        # Filter non-Python files
        valid_files = []
        valid_patches = []
        for i in range(len(files)):
            if files[i].endswith(".py"):
                valid_files.append(files[i])
                valid_patches.append(patches[i])
        
        # Filter files that exist in the graph
        nfiles = currgraph.filter(
            lambda path, type: path in valid_files and type == nodetypedict["python file"],
            input_columns=["path", "type"]
        )
        
        if len(nfiles) == 0:
            # Return empty list instead of numpy array
            return {"files": [], "timestamp": time}
        
        #print(f"len nfiles {time}", len(nfiles))
        file_ids = nfiles["id"]  # This is a list of integers
        #print(f"len file_ids {time}", len(file_ids))
        
        return {"files": file_ids, "timestamp": time}

    commitpulldata = embpulldata.map(committransform)
    
    
    # Select columns and filter empty files
    data = commitpulldata.select_columns([
        "files", "timestamp", 
        "bug_desc_emb", "expected_behavior_emb", "actual_behavior_emb", 
        "reproduce_emb", "require_emb", "code_emb", 
        "error_trace_emb", "error_statement_emb", "issue_title_emb"
    ])
    
    
    #data = data.filter(lambda files_list: len(files_list) > 0, input_columns=["files"])    
    # Convert to torch format ONLY HERE at the end
    data = data.with_format("torch")
    data.save_to_disk(f"taskdata/{REPO_NAME}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    args = parser.parse_args()    
    REPO_NAME = args.name
    main(REPO_NAME)