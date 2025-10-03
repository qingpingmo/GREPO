import os
import torch
import pickle
import subprocess
from datasets import load_from_disk
import argparse
import sys
import threading
import queue
from modelscope import AutoTokenizer, AutoModel
from typing import Dict, List, Any
from torch import Tensor

class RepoHandler:
    def __init__(self, repo_name, repo_owner=None, token="github_pat_xxx", 
                 keep="merged", keep_wo_issue=False, keep_non_main_branch=False, cpu=1, save_path="pulldata"):
        self.repo = repo_name
        self.repo_owner = repo_owner
        self.token = token
        self.keep = keep
        self.keep_wo_issue = keep_wo_issue
        self.keep_non_main_branch = keep_non_main_branch
        self.cpu = cpu
        self.save_path = save_path

    def get_graph(self):
        path = f"pyggraph/{self.repo}.timed.pt"
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")
        else:
            raise FileNotFoundError(f"Graph file not found: {path}")

    def get_label(self):
        path = f"taskdata/{self.repo}"
        if os.path.exists(path):
            return load_from_disk(path)
        else:
            raise FileNotFoundError(f"Label folder not found: {path}")

    def get_sha(self):
        path = f"pulldata/{self.repo}_sha_path.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"SHA file not found: {path}")

    def get_commit(self,device="cuda",batch_size=4):
        if not os.path.exists("unconditional_label"):
            os.makedirs("unconditional_label")
        unlabeled_path = f"unconditional_label/{self.repo}.pt"
        if os.path.exists(unlabeled_path):
            emb = torch.load(unlabeled_path, map_location="cpu")
        else:
            from commit_utils import CommitDAGAnalyzer
            import git
            from sentence_transformers import SentenceTransformer
            analyzer = CommitDAGAnalyzer(repo_name=self.repo)
            sha_path = analyzer.get_longest_path()
            repo = git.Repo(f"repos/{self.repo}")

            def b2s(msg):
                return str(msg, encoding="utf-8") if isinstance(msg, bytes) else msg

            msgs = [b2s(repo.commit(sha).message) for sha in sha_path[1:]]
            #model = SentenceTransformer("all-mpnet-base-v2", device="cuda")
            #emb = model.encode(msgs, convert_to_tensor=True)
            #model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-8B', local_files_only=True if os.path.exists('Qwen/Qwen3-Embedding-8B') else False).eval().to(device)
            #tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-8B', padding_side='left', local_files_only=True if os.path.exists('Qwen/Qwen3-Embedding-8B') else False)
            from vllm import LLM
            model = LLM(
                    model="Qwen3-Embedding-8B",
                    task="embed",
                    enforce_eager=True,

                )
            tokenizer = AutoTokenizer.from_pretrained(
                'Qwen/Qwen3-Embedding-8B',
                padding_side='left',
                trust_remote_code=True,  
                local_files_only=os.path.exists('Qwen/Qwen3-Embedding-8B')
            )
            max_length = 16384
            prompt_token_id_commit = tokenizer(
                msgs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prompt_commit = tokenizer.batch_decode(prompt_token_id_commit["input_ids"], skip_special_tokens=True)
            emb_outputs_commit = model.embed(prompt_commit)
            if hasattr(emb_outputs_commit[0], "embedding"):
                emb_commit = [out.embedding for out in emb_outputs_commit]
            elif hasattr(emb_outputs_commit[0], "hidden_states"):
                emb_commit = [out.hidden_states for out in emb_outputs_commit]
            elif hasattr(emb_outputs_commit[0], "outputs") and hasattr(emb_outputs_commit[0].outputs, "embedding"):
                emb_commit = [out.outputs.embedding for out in emb_outputs_commit]
            elif hasattr(emb_outputs_commit[0], "outputs") and hasattr(emb_outputs_commit[0].outputs, "hidden_states"):
                emb_commit = [out.outputs.hidden_states for out in emb_outputs_commit]
            else:
                raise ValueError("Cannot extract embedding from model.embed output; please check the output structure.")

            emb = torch.tensor([e.tolist() if hasattr(e, "tolist") else list(e) for e in emb_commit])


            def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
            
                left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                if left_padding:
                    return last_hidden_states[:, -1]
                else:
                    sequence_lengths = attention_mask.sum(dim=1) - 1
                    batch_size = last_hidden_states.shape[0]
                    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

            def get_embeddings(input_texts: List[str], batch_size: int = 4,model = None, tokenizer = None, max_length = 2048) -> Tensor:

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

            
            
            
            #emb = get_embeddings(msgs, batch_size = batch_size, model = model, tokenizer = tokenizer, max_length = max_length)
            
            emb = emb.cpu()
            torch.save(emb, unlabeled_path)
        return emb
    
    def run_command(self, cmd, shell=False):
        
        if isinstance(cmd, list):
            print(f"\n\033[1;34mRunning command:\033[0m {' '.join(cmd)}")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  
                universal_newlines=True
            )
        else:
            print(f"\n\033[1;34mRunning command:\033[0m {cmd}")
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  
                universal_newlines=True
            )
        
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        
        return_code = process.poll()
        return return_code == 0
    
    def build_graph(self):
        try:
            
            repo_output_dir = os.path.join("savedata", "repos", self.repo)
            if not os.path.exists(repo_output_dir) or not os.listdir(repo_output_dir):
                repo_analysis_cmd = f"python RepoAnalysis.py {self.repo}"
                if not self.run_command(repo_analysis_cmd, shell=True):
                    return False
            else:
                print(f"Skipping RepoAnalysis.py for {self.repo}, output {repo_output_dir} already exists")
            
            
            intermediate_file = os.path.join("pyggraph", f"{self.repo}.pt")
            final_file = os.path.join("pyggraph", f"{self.repo}.timed.pt")
            
            
            if not os.path.exists(final_file):
                build_graph_cmd = f"python buildGraphTemporal.py {self.repo}"
                if not self.run_command(build_graph_cmd, shell=True):
                    return False
            else:
                print(f"Skipping buildGraphTemporal.py for {self.repo}, output {final_file} already exists")
            
            return True
        except Exception as e:
            print(f"Error building graph for {self.repo}: {e}")
            return False
    
    def build_label(self):
        if not self.repo_owner:
            raise ValueError("repo_owner must be provided to build labels")
        
        try:
            
            github_output = os.path.join(self.save_path, self.repo)
            if os.path.exists(github_output) and os.path.isdir(github_output):
                
                dataset_files = os.listdir(github_output)
                if dataset_files and any(file.endswith('.arrow') for file in dataset_files):
                    print(f"Skipping GithubRepoAnalysis.py for {self.repo}, output already exists, files: {github_output}/{dataset_files}")
                else:
                    
                    github_cmd = [
                        "python", "GithubRepoAnalysis.py",
                        "-o", self.repo_owner,
                        "-n", self.repo,
                        "-c", str(self.cpu),
                        "-t", self.token,
                        "--keep", self.keep,
                        "--save_path", self.save_path
                    ]
                    if self.keep_wo_issue:
                        github_cmd.append("--keep_wo_issue")
                    if self.keep_non_main_branch:
                        github_cmd.append("--keep_non_main_branch")
                    if not self.run_command(github_cmd):
                        return False
            else:
                
                github_cmd = [
                    "python", "GithubRepoAnalysis.py",
                    "-o", self.repo_owner,
                    "-n", self.repo,
                    "-c", str(self.cpu),
                    "-t", self.token,
                    "--keep", self.keep,
                    "--save_path", self.save_path
                ]
                if self.keep_wo_issue:
                    github_cmd.append("--keep_wo_issue")
                if self.keep_non_main_branch:
                    github_cmd.append("--keep_non_main_branch")
                if not self.run_command(github_cmd):
                    return False
            
            
            issue_output = os.path.join("pulldata", f"{self.repo}_postpro")
            if os.path.exists(issue_output) and os.path.isdir(issue_output):
                
                issue_files = os.listdir(issue_output)
                if issue_files and any(file.endswith('.arrow') for file in issue_files):
                    print(f"Skipping issue_postpro.py for {self.repo}, output {issue_output} already exists")
                else:
                    issue_cmd = f"python issue_postpro.py {self.repo}"
                    if not self.run_command(issue_cmd, shell=True):
                        return False
            else:
                issue_cmd = f"python issue_postpro.py {self.repo}"
                if not self.run_command(issue_cmd, shell=True):
                    return False
            
            
            task_output = os.path.join("taskdata", self.repo)
            if os.path.exists(task_output) and os.path.isdir(task_output):
                
                task_files = os.listdir(task_output)
                if task_files and any(file.endswith('.arrow') for file in task_files):
                    print(f"Skipping Taskdata.py for {self.repo}, output {task_output} already exists")
                else:
                    taskdata_cmd = f"python Taskdata.py {self.repo}"
                    if not self.run_command(taskdata_cmd, shell=True):
                        return False
            else:
                taskdata_cmd = f"python Taskdata.py {self.repo}"   #beforeit:export HF_ENDPOINT=https://hf-mirror.com 
                if not self.run_command(taskdata_cmd, shell=True):
                    return False
            
            return True
        except Exception as e:
            print(f"Error building labels for {self.repo}: {e}")
            return False
    
    def build_dataset(self):
        print(f"\n\033[1;32mBuilding dataset for {self.repo}...\033[0m")
        
        
        result_queue = queue.Queue()
        
        
        def run_graph():
            try:
                print("\n\033[1;33m=== Starting Graph Build ===\033[0m")
                success = self.build_graph()
                result_queue.put(("graph", success))
            except Exception as e:
                print(f"\033[1;31mGraph build failed with exception: {e}\033[0m")
                result_queue.put(("graph", False))
        
        def run_label():
            try:
                print("\n\033[1;33m=== Starting Label Build ===\033[0m")
                success = self.build_label()
                result_queue.put(("label", success))
            except Exception as e:
                print(f"\033[1;31mLabel build failed with exception: {e}\033[0m")
                result_queue.put(("label", False))
        
        
        graph_thread = threading.Thread(target=run_graph)
        label_thread = threading.Thread(target=run_label)
        
        graph_thread.start()
        label_thread.start()
        
        
        graph_thread.join()
        label_thread.join()
        
        
        results = {}
        while not result_queue.empty():
            task, success = result_queue.get()
            results[task] = success
        
        
        graph_success = results.get("graph", False)
        label_success = results.get("label", False)
        
        if graph_success and label_success:
            print(f"\n\033[1;32mDataset build complete for {self.repo}\033[0m")
            return True
        else:
            if not graph_success:
                print("\033[1;31mGraph build failed!\033[0m")
            if not label_success:
                print("\033[1;31mLabel build failed!\033[0m")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repository Data Handler")
    parser.add_argument("--repo_name", type=str, required=True, help="Name of the repository")
    parser.add_argument("-o", "--repo_owner", type=str, help="The owner of the repository")
    parser.add_argument("-t", "--token", type=str, default="github_pat_xxx", 
                        help="The access token for the github")
    parser.add_argument("--keep", type=str, default="merged", choices=["merged", "closed", "approved", "all"], 
                        help="The pull requests to keep")
    parser.add_argument("--keep_wo_issue", action="store_true", help="Whether to include the issues")
    parser.add_argument("--keep_non_main_branch", action="store_true", 
                        help="Whether to only include the pull requests in the main branch")
    parser.add_argument("-c", "--cpu", type=int, default=1, help="The number of cpus to use")
    parser.add_argument("--save_path", type=str, default="pulldata", help="The path to save the pull data")
    parser.add_argument("--build", action="store_true", help="Build the dataset from scratch")
    parser.add_argument("--build_graph", action="store_true", help="Build the graph from scratch")
    parser.add_argument("--build_label", action="store_true", help="Build the label from scratch")
    
    args = parser.parse_args()

    
    handler = RepoHandler(
        repo_name=args.repo_name,
        repo_owner=args.repo_owner,
        token=args.token,
        keep=args.keep,
        keep_wo_issue=args.keep_wo_issue,
        keep_non_main_branch=args.keep_non_main_branch,
        cpu=args.cpu,
        save_path=args.save_path
    )
    
    
    if args.build:
        if args.build_graph:
            success = handler.build_graph()
            if not success:
                print("\033[1;31mGraph build failed!\033[0m")
                sys.exit(1)
        
        elif args.build_label:
            success = handler.build_label()
            if not success:
                print("\033[1;31mLabel build failed!\033[0m")
                sys.exit(1)
        
        else:
            success = handler.build_dataset()
        
        if not success:
            print("\033[1;31mDataset build failed!\033[0m")
            sys.exit(1)
    else:
        
        print("\n\033[1;32mLoading existing data...\033[0m")
        try:
            graph_data = handler.get_graph()
            label_data = handler.get_label()
            sha_data = handler.get_sha()
            commit_emb = handler.get_commit()

            print(f"\nGraph data loaded with type: {type(graph_data)}")
            print(f"Label data loaded with type: {type(label_data)}")
            if hasattr(label_data, 'features'):
                print(f"Label features: {label_data.features}")
            print(f"SHA data loaded with type: {type(sha_data)}")
            print(f"Commit embeddings shape: {commit_emb.shape}")
        except Exception as e:
            print(f"\033[1;31mError loading data: {e}\033[0m")
            sys.exit(1)