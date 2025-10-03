from typing import Dict, List, Tuple, Literal, Self
import os.path as osp
import os
import git
from git.diff import Diff
import subprocess
import argparse
import datasets
from commit_utils import CommitDAGAnalyzer
import sys
import json
REPO_NAME = "scipy"
repo_path = f"repos/{REPO_NAME}"
dataset = datasets.Dataset.load_from_disk(f"savedata/repos/{REPO_NAME}")

MAIN_BRANCH_NAME = "main"
try:
    print(subprocess.check_output(f"cd {repo_path}; git checkout -f {MAIN_BRANCH_NAME}", shell=True))
except:
    MAIN_BRANCH_NAME = "master"
    print(subprocess.check_output(f"cd {repo_path}; git checkout -f {MAIN_BRANCH_NAME}", shell=True))
analyzer = CommitDAGAnalyzer(REPO_NAME)
sha_path = analyzer.get_longest_path()

commit2time = {sha: i for i, sha in enumerate(sha_path)}
commit2time["none"] = sys.maxsize

def converttime(indict):
    ret = {}
    ret["start_time"] = commit2time[indict["start_commit"]]
    ret["end_time"] = commit2time[indict["end_commit"]]
    return ret

dataset = dataset.map(converttime, num_proc=32)


repo = git.Repo(repo_path)
for sha in sha_path[1:]:
    t = commit2time[sha] - 1
    commit_msg = repo.commit(sha).message
    print("commit_msg:", commit_msg)
    tgraph = []    
    rootnode = dataset[0]
    def visit(node):
        tgraph.append(node)
        for child in node["contain"]:
            if dataset[child]["start_time"] <= t < dataset[child]["end_time"]:
                visit(dataset[child])
    
    visit(rootnode)
    print("all repo")
    for node in tgraph:
        if node["type"] == 2:
            code = json.loads(node["attr"])["code"]
            path = node["path"]
            print(path, code)

    print("changed")
    for node in tgraph:
        if node["type"] == 2:
            if node["end_time"] == t+1:
                path = node["path"]
                print(path)