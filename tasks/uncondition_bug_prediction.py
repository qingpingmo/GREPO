import os
import git
import re
import datasets
import json

from typing import Iterator, TypedDict, Self
from datetime import datetime
from commit_utils import CommitDAGAnalyzer

from tasks.patch_utils import parse_commit
from tasks.node_utils import Nodes
from utils import logger

class TagDict(TypedDict):
    tag: str
    commit_sha: str
    commit_time: datetime
    
class UnconditionBugPredictionSample(TypedDict):
    tag: str # the tag name, it is the start of the duration. The model is expected to predict all bug fix from the tag until the next tag.
    commit_sha: str # the commit sha of the tag. This tag is labeled just AFTER the tag, so the tag introduced by this commit has been included in this tag and should NOT be predicted.
    text_graph: str # A json dumps string of a diction[str, int], where the key is the file::function/class_name and the value is the label of the node. The label is an int, where 0 means Untorched, 1 means Bug and 2 means Feature.

class NodeLabel:
    Untorched = 0
    Bug = 1
    Feature = 2
    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError("value must be int.") 
        if value not in (0, 1, 2):
            raise ValueError("value must be 0, 1 or 2.")
        self._value = value
    def __repr__(self) -> str:
        return str(self._value)
    def __str__(self) -> str:
        return str(self._value)
    def __int__(self) -> int:
        return self._value
    def update(self, new_value: int | Self):
        """Find the oldest non-untorched commit as its label."""
        if isinstance(new_value, self.__class__):
            new_value = int(new_value)
        if not isinstance(new_value, int):
            raise TypeError("value must be int.")
        if new_value not in (0, 1, 2):
            raise ValueError("value must be 0, 1 or 2.")
        if self._value in (1, 2) and new_value == 0:
            # do nothing
            return
        else:
            self._value = new_value
    
    
RECOGNIZED_DEV_TYPES = (
    "API",
    "BENCH",
    "BLD",
    "BUG",
    "CI",
    "DEP",
    "DOC",
    "ENH",
    "FIX",
    "MAINT",
    "MNT",
    "PERF",
    "REFACTOR",
    "RELEASE",
    "REL",
    "REV",
    "STYLE",
    "STY",
    "TEST",
    "TST",
    "TYP",
    "WIP",
)
BUG_DEV_TYPES = (
    "BUG",
    "FIX",
    "DOC",
    "TYP"
)
assert all(
    dev_type in RECOGNIZED_DEV_TYPES for dev_type in BUG_DEV_TYPES
), "All bug development types must be in the recognized development types."
    

class UnconditionBugPredictionTask:
    def __init__(self, repo: str, filter_tags: re.Pattern | bool = False, first_commit: str | None = None, last_commit: str | None = None, first_tag: str | None = None, last_tag: str | None = None, repo_root_path: str = "repos", pr_data_path: str = "pulldata_wo_issue", commitdata_path: str = "savedata/repos/", commit_dag_analyzer_path: str = "commit_processed"):
        """
        Args:
            repo (str): The name of the repository.
            filter_tags (re.Pattern | bool): Filter the tags by a regex pattern. If `True`, only keep the vx.x.x or x.x.x tags. If `False`, no filtering.
            first_commit (str | None): The first commit to start the data duration. If `None`, use the first commit in the repository. This commit itself is included in the data duration.
            last_commit (str | None): The last commit to end the data duration. If `None`, use the last commit in the repository. This commit itself is included in the data duration.
            first_tag (str | None): The first tag to start the data duration. If `None`, it is begin from an empty repo. This commits after the tag are expected to be predicted.
            last_tag (str | None): The last tag to end the data duration. If `None`, include the newest commit in the repo. This commits before the tag are expected to be predicted.
        """
        self.repo = repo
        self.filter_tags = filter_tags
        self.first_commit = first_commit
        self.last_commit = last_commit
        if first_tag is not None:
            if self.first_commit is not None:
                logger.warning("Both first_tag and first_commit are provided, using first_tag as the start of the data duration.")
            self.first_commit = self.find_commit_from_tag_name(first_tag)
        if last_tag is not None:
            if self.last_commit is not None:
                logger.warning("Both last_tag and last_commit are provided, using last_tag as the end of the data duration.")
            self.last_commit = self.find_commit_from_tag_name(last_tag)
        self.commit_dag_analyzer = CommitDAGAnalyzer(repo_name=repo, repo_root_path=repo_root_path, processed_path=commit_dag_analyzer_path)
        self.git_longest_path = self.commit_dag_analyzer.get_longest_path() # We simplify the repo history to a longest path to make the history linear.
        if self.first_commit is not None and self.first_commit not in self.git_longest_path:
            raise ValueError(f"first_commit {self.first_commit} is not in the longest path of the repo.")
        if self.last_commit is not None and self.last_commit not in self.git_longest_path:
            raise ValueError(f"last_commit {self.last_commit} is not in the longest path of the repo.")
        self.repo_root_path = repo_root_path
        self.local_repo_path = os.path.join(repo_root_path, repo)
        self.local_pr_data_path = os.path.join(pr_data_path, repo)
        self.nodes = self.load_nodes(os.path.join(commitdata_path, self.repo))
        
    def load_nodes(self, load_dataset_path: str) -> Nodes:
        """
        Load the nodes from the dataset.
        """
        return Nodes(self.repo, load_dataset_path)
    
    def find_commit_from_tag_name(self, tag_name: str) -> str:
        """
        Find the commit sha from the tag name.
        """
        local_repo = git.Repo(self.local_repo_path)
        tags = local_repo.tags
        for tag in tags:
            if tag.name == tag_name:
                return tag.commit.hexsha
        raise ValueError(f"Tag {tag_name} not found in the repository.")
    
    def _retrieve_pr_number_from_commit_message(self, message: str) -> int | None:
        """
        Retrieve the PR number from the commit message.
        """
        pr_number = re.search(r"\(#(\d+)\)", message)
        if pr_number is not None:
            return int(pr_number.group(1))
        return None
    
    def _retrieve_dev_type_from_commit_message(self, message: str) -> str | None:
        """
        Retrieve the development type from the commit message.
        """
        first_word = message.split()[0].replace(":", "").upper()
        if first_word in RECOGNIZED_DEV_TYPES:
            return first_word
        return None
    
    # TODO: if no dev type is found in message, we can check by the label or linked issue.
    def _is_bugfix_commit(self, message: str) -> bool:
        """
        Check if the commit is a bugfix commit.
        """
        dev_type = self._retrieve_dev_type_from_commit_message(message)
        return dev_type in BUG_DEV_TYPES
    
    def retrieve_info_from_commit(self, commit: git.Commit):
        """
        Retrieve information from the commit.
        """
        message = str(commit.message)
        # find the pr number in the commit message
        pr_number = self._retrieve_pr_number_from_commit_message(message)
        # develop_type = 
        
        
        commit_info = {
            "pr_number": pr_number,
            
        }
        return commit_info
    
    def get_pointtimes(self, filter_: bool | re.Pattern = False) -> list[TagDict]:
        """
        Get the commit sha of tags. From this commit, the version is labeled by the tag. It means when you checkout to the tag, it is equivalent to checkout to the commit sha.
        """
        local_repo = git.Repo(self.local_repo_path)
        tags = local_repo.tags
        
        if filter_ is True:
            # only keep the vx.x.x or x.x.x tags
            filter_pattern = re.compile(r"^v?\d+(\.\d+)*$")
        elif isinstance(filter_, re.Pattern):
            filter_pattern = filter_
        elif filter_ is False:
            filter_pattern = None
        else:
            raise ValueError("filter must be a bool or a regex pattern.")
        
        if filter_pattern is not None:
            tags = [tag for tag in tags if filter_pattern.match(tag.name)]
        
        tags_dicts: list[TagDict] = []
        for tag in tags:
            commit = tag.commit
            commit_sha = commit.hexsha
            commit_time = commit.committed_datetime
            # print("debug", commit_sha, self.git_longest_path)
            if commit_sha not in self.git_longest_path:
                logger.warning(f"Tag {tag.name} is not in the longest path of the repo, now we just skip it.")
                tags_dicts.append({
                    "tag": tag.name,
                    "commit_sha": commit_sha,
                    "commit_time": commit_time,
                })
            
        tags_dicts = [tag for tag in tags_dicts if self.commit_dag_analyzer.is_commit_in_range(tag["commit_sha"], self.first_commit, self.last_commit, include_left=False, include_right=True)]
        # Here we include the right side and exclude the left side to make sure if we use the tag as the start of the duration, the commit itself is not included in the duration.
            
        # rerange the tags by time, from the earliest to the latest
        # NOTE: although the commit time is not accurately record the order of the commits, the tags are gapped by a large time interval, so we can use the commit time to sort the tags.
        tags_dicts.sort(key=lambda x: x["commit_time"], reverse=False)
        return tags_dicts
    
    def _concat_file_name_and_funccls_name(self, file: str, funccls_name: str) -> str:
        return file + "::" + funccls_name
    
    def _split_file_name_and_funccls_name(self, file_funccls_name: str) -> tuple[str, str]:
        file, funccls_name = file_funccls_name.split("::")
        return file, funccls_name
            
    def find_changes(self, commit: git.Commit, all_changes: dict[str, NodeLabel]) -> dict[str, NodeLabel]:
        """
        The inner logic to accumulate changes during a duration. 
        Given a git.Commit object, find the changes in the commit and then update the all_changes dict.
        The all_changes dict is a dict of file_funccls_name -> NodeLabel.
        """
        base_commit = self.commit_dag_analyzer.get_base_commit(commit.hexsha)
        if base_commit is not None:
            base_commit = git.Repo(self.local_repo_path).commit(base_commit)
        else:
            base_commit = None
        commit_type = NodeLabel.Bug if self._is_bugfix_commit(commit.message if isinstance(commit.message, str) else commit.message.decode('utf-8')) else NodeLabel.Feature # type: ignore

        file_funccls_names = parse_commit(
            repo_name = self.repo,
            base_commit = base_commit,
            commit = commit, 
            repo_root_path = self.repo_root_path,
            detailed = True,
        )
        # print(file_funccls_names)
        old_file_funccls_names = [item[0] for item in file_funccls_names if len(item)>0 and item[0] is not None]
        
        for file, funccls_name in old_file_funccls_names:
            file_funccls_name = self._concat_file_name_and_funccls_name(file, funccls_name)
            if file_funccls_name not in all_changes:
                all_changes[file_funccls_name] = NodeLabel(commit_type)
            else:
                all_changes[file_funccls_name].update(commit_type)
                
        return all_changes
            
        
    def project_changes_to_start_commit(self, first_commit: git.Commit, all_changes: dict[str, NodeLabel]) -> dict[str, NodeLabel]:
        """
        When we meet the start commit of a data duration, we need to project all changes accumulated from the last tag to the start commit into the nodes at this time.
        
        Return:
            A dictionary of file_funccls_name -> NodeLabel. The dict contain and only contain all nodes where the given `first_commit` is in whose lifetime.
        """
        # first find all nodes in first commit
        nodes_at_start_commit: datasets.Dataset = self.nodes.select_nodes_by_commit(commit_sha=first_commit.hexsha)
        
        file_names: list[str] = self.nodes.find_dirfile_name(subset=nodes_at_start_commit)
        funccls_names: list[str] = self.nodes.find_funccls_name(subset=nodes_at_start_commit)
        
        all_nodes_names_at_start_commit: list[str] = [self._concat_file_name_and_funccls_name(file_name, funccls_name) for file_name, funccls_name in zip(file_names, funccls_names)]
        # by default, the nodes are labeled as NodeLabel[Untorched]
        node_labels: dict[str, NodeLabel] = {node_name: NodeLabel(NodeLabel.Untorched) for node_name in all_nodes_names_at_start_commit}
        
        # then, all changes are recorded the oldest non-untorched commit as its label
        for file_funccls_name, label in all_changes.items():
            node_labels[file_funccls_name].update(label)
        return node_labels
    
    def process(self) -> list[UnconditionBugPredictionSample]:
        print("start")
        tags = self.get_pointtimes(filter_=self.filter_tags)
        last_tag = tags[-1]
        data_samples: list[UnconditionBugPredictionSample] = []
        local_repo = git.Repo(self.local_repo_path)
        
        all_changes = {}
        commit_iter = self.git_longest_path
        commit_iter.reverse()
        for commit in commit_iter:
            if not self.commit_dag_analyzer.is_commit_in_range(commit, self.first_commit, self.last_commit, include_left=True, include_right=True):
                # here we include the left to make sure the project can be done. But the right side is not included in the text_graph.
                continue
            if commit == last_tag["commit_sha"]:
                # the tag is AFTER the commit, so the change introduced by the commit should not be included in the tag, but included in the last tag.
                data_sample = self.project_changes_to_start_commit(first_commit=local_repo.commit(commit), all_changes=all_changes)
                # change NodeLabel to int to facilitate the json.dumps
                data_sample_int = {
                    file_funccls_name: int(label) for file_funccls_name, label in data_sample.items()
                }
                data_samples.append({
                    "tag": last_tag["tag"], 
                    "commit_sha": last_tag["commit_sha"], 
                    "text_graph": json.dumps(data_sample_int)
                    })
                last_tag = tags.pop()
            if self.first_commit is None or self.commit_dag_analyzer.is_ancestor(self.first_commit, commit, strict=True):
                all_changes = self.find_changes(local_repo.commit(commit), all_changes)
        return data_samples
    
    def save(self, save_path: str):
        from datasets import Dataset
        data_samples = self.process()
        dataset = Dataset.from_list(data_samples)
        os.makedirs(save_path, exist_ok=True)
        dataset.save_to_disk(save_path)
                

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()
    task = UnconditionBugPredictionTask(args.repo, True, None, None, None, None, )
    task.save(save_path='uncondition_pred_dataset')
