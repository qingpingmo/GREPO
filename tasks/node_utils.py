from datasets import Dataset, load_from_disk

from commit_utils import CommitDAGAnalyzer
from tasks.patch_utils import parse_commit
from tasks.patch_utils import DetailedClassFunctionInfo
import os
import subprocess
import git
from git import NULL_TREE
import json

from typing import Sequence, Literal

from utils import logger

class CommitParser:
    def __init__(self, repo_name: str, repo_root_path: str = "repos"):
        self.repo_name = repo_name
        self.repo_root_path = repo_root_path
    
    def parse_change_relation_of_func_cls(self, commit: git.Commit) -> dict[str | Literal[git.DiffConstants.NULL_TREE], list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]]:
        
        parents: Sequence[git.Commit] | list[None] = commit.parents
        if len(parents) == 0:
            parents = [None]
        # get the file change relation of the commit
        files_change_from_parents: dict[str | Literal[git.DiffConstants.NULL_TREE], list[tuple[None | DetailedClassFunctionInfo, None | DetailedClassFunctionInfo]]] = dict()
        for parent in parents:
            old_new_nodes = parse_commit(
                repo_name = self.repo_name,
                base_commit=parent,
                commit=commit,
                repo_root_path=self.repo_root_path,
                detailed=True,
            )
            from typing import cast
            parent_name = cast(str | Literal[git.DiffConstants.NULL_TREE], getattr(parent, "hexsha", NULL_TREE))
            files_change_from_parents[parent_name] = old_new_nodes    
        return files_change_from_parents

class Nodes:
    def __init__(self, repo_name: str, load_dataset_path: str, repo_root_path: str = "repos", processed_commit_dag_analyzer_path: str = 'commit_processed'):
        """
        Args:
            load_dataset_path (str): Path to the dataset directory. It is a dataset contain all nodes of the repo.
        """
        self._nodes = load_from_disk(load_dataset_path)
        self.commit_dag_analyzer = CommitDAGAnalyzer(repo_name=repo_name, processed_path=processed_commit_dag_analyzer_path, repo_root_path=repo_root_path)
    
    @classmethod
    def parse_from_repo(cls, repo_name: str, save_dataset_path: str, repo_root_path: str = "repos", repo_owner: str | None = None, processed_commit_dag_analyzer_path: str = 'commit_processed'):
        """
        Download the repo from github if needed, and collect commit data on the main branch.
        """
        raise NotImplementedError("This function is not implemented yet.")
        # 1. Download the repo from github if needed
        os.makedirs(repo_root_path, exist_ok=True)
        repo_path = os.path.join(repo_root_path, repo_name)
        if not os.path.exists(repo_path):
            if repo_owner is None:
                raise ValueError("repo_owner should be specified when the repo is not downloaded.")
            # Download the repo from github
            logger.info(f"Start downloading repo {repo_owner}/{repo_name} from github...")
            subprocess.check_output(f"cd repos; git clone https://github.com/{repo_owner}/{repo_name}.git; cd ..", shell=True)
        # 2. Checkout the main branch
        try:
            subprocess.check_output(f"cd {repo_path}; git checkout main", shell=True)
            main_branch_name = "main"
        except subprocess.CalledProcessError:
            subprocess.check_output(f"cd {repo_path}; git checkout master", shell=True)
            main_branch_name = "master"
            
        repo = git.Repo(repo_path)
        commit_dag_analyzer = CommitDAGAnalyzer(repo_name=repo_name, processed_path=processed_commit_dag_analyzer_path, repo_root_path=repo_root_path)
        
        # 3. Get the commit data
        commits_sha_dict = {c.hexsha: c for c in repo.iter_commits()}
        commits = [commits_sha_dict[c] for c in commit_dag_analyzer.topo_sort()]
        
        logger.info(f"Total {len(commits)} commits in the repo {repo_name}.")
        
        text_graph = []
        for commit in commits:
            ...
            
        
        
        
    @property
    def nodes(self) -> Dataset:
        return self._nodes
    
    @property
    def dataset(self) -> Dataset:
        return self._nodes
        
    def select_nodes_by_commit(self, commit_sha: str) -> Dataset:
        """
        Return a subset of the dataset that contains all nodes at the time of the given commit timestamp.
        """
        # Filter the dataset to include only nodes that are present at the given commit timestamp

        filtered_nodes = self.nodes.filter(lambda version, endversion: self.commit_dag_analyzer.is_commit_in_range(commit_sha, start=version, end=endversion, include_left=True, include_right=False), input_columns=["version", "endversion"])
        return filtered_nodes
    
    def find_dirfile_name(self, subset: Dataset | None = None) -> list[str]:
        if subset is None:
            subset = self.nodes
        # Get the directory and file names from the dataset
        dirfile_names = []
        for d in subset:
            if d["type"] in ["directory", "file", "Python file"]:
                dirfile_names.append(d["name"])
            elif d["type"] in ["class def", "function def"]:
                dirfile_names.append(json.loads(d["attr"])["filename"])
            else:
                raise ValueError(f"Unknown type {d['type']}")
        return dirfile_names
    
    def find_funccls_name(self, subset: Dataset | None = None) -> list[str]:
        if subset is None:
            subset = self.nodes
        # Get the function and class names from the dataset
        funccls_names = []
        for d in subset:
            if d["type"] in ["class def", "function def"]:
                funccls_names.append(d["name"])
            else:
                funccls_names.append("")
        return funccls_names