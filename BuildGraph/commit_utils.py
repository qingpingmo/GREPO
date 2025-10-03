import git
from collections import deque
from typing import List, Generator
import os
import numpy as np
import time
from typing import Sequence
from utils import logger

class CommitDAGAnalyzer:
    """
    A class to analyze the commit history of a git repository and build a directed acyclic graph (DAG) of commits. Provide some functions to (1) the commit idx and topo order, (2) get the ancestor relation between commits.
    """
    
    def __init__(self, repo_name: str, repo_root_path: str = './repos/', processed_path: str = './commit_processed/'):
        if not os.path.exists(processed_path):
            os.makedirs(processed_path)
        self.repo_path = os.path.join(repo_root_path, repo_name)
        self.repo = git.Repo(self.repo_path)
        self.commits = list(self.repo.iter_commits())
        self.n = len(self.commits)
        self._commit_to_idx = {commit.hexsha: i for i, commit in enumerate(self.commits)}
        self._idx_to_commit = {i: commit.hexsha for i, commit in enumerate(self.commits)}
        self._build_graph()
        save_path = os.path.join(processed_path, f"{repo_name}.npy")
        self._build_ancestor_matrix()
        self._build_longest_path()
        self.repo.close()

    def _build_graph(self):
        self.graph = [[] for _ in range(self.n)]
        for i, commit in enumerate(self.commits):
            for parent in commit.parents:
                assert parent.hexsha in self._commit_to_idx
                parent_idx = self._commit_to_idx[parent.hexsha]
                self.graph[parent_idx].append(i)
    
    def _build_ancestor_matrix(self) -> None:
        matrix = [None] * self.n
        
        topo_order = [self._commit_to_idx[x] for x in self.topo_order()]
        assert len(topo_order) == self.n, "Graph is not a DAG"
        topo_order.reverse()
        
        for i in topo_order:
            row = np.zeros(self.n, dtype=bool)
            row[i] = True
            for neighbor in self.graph[i]:
                row = row | matrix[neighbor]
            matrix[i] = row
            
        assert all(m is not None for m in matrix), "Matrix is not built correctly"
        self.matrix = np.vstack(matrix)

    def _save_matrix_to_file(self, path: str):
        np.save(path, self.matrix)

    def _load_matrix_from_file(self, path: str):
        self.matrix = np.load(path)
        
    def commit_to_idx(self, commit: str) -> int:
        if commit not in self._commit_to_idx:
            raise ValueError(f"Commit {commit} not found in the repository.")
        return self._commit_to_idx[commit]
    
    def idx_to_commit(self, idx: int) -> str:
        if idx not in self._idx_to_commit:
            raise ValueError(f"Index {idx} not found in the repository.")
        return self._idx_to_commit[idx]
    
    def is_ancestor(self, commit_a: str, commit_b: str, *, strict: bool = False) -> bool:
        if commit_a == commit_b:
            return not strict
        idx_a, idx_b = self.commit_to_idx(commit_a), self.commit_to_idx(commit_b)
        assert self.matrix is not None, "Matrix is not built yet"
        return self.matrix[idx_a][idx_b]
        
    def is_commit_in_range(self, commit: str, start: str | None, end: str | None, *, include_left: bool = True, include_right: bool = True):
        """
        Alias for is_ancestor(start, commit) and is_ancestor(commit, end). Note that the function is different to is_commit_in_lifespan. Because of the siblings, the lifespan could be larger than the range.
        """
        if start is None and end is None:
            return True
        elif start is None:
            assert end is not None
            return self.is_ancestor(commit, end, strict = not include_right)
        elif end is None:
            return self.is_ancestor(start, commit, strict = not include_left)
        else:
            return self.is_ancestor(start, commit, strict = not include_left) and self.is_ancestor(commit, end, strict = not include_right)
        
    def is_commit_in_lifespan(self, commit: str, start: str | None, ends: Sequence[str] | str | None, *, include_left: bool = True, include_right: bool = True):
        """
        Check if commit is in the lifespan of start and ends at ends.
        Alias for is_ancestor(start, commit) and not is_ancestor(end, commit) for each end in ends. Note that the function is different to is_commit_in_range. Because of the siblings, the lifespan could be larger than the range.
        """
        if isinstance(ends, str):
            ends = [ends]
        if start is None and ends is None:
            return True
        elif start is None:
            assert ends is not None
            return all([not self.is_ancestor(end, commit, strict = include_right) for end in ends])
        elif ends is None:
            return self.is_ancestor(start, commit, strict = not include_left)
        else:
            assert ends is not None
            return self.is_ancestor(start, commit, strict = not include_left) and all([not self.is_ancestor(end, commit, strict = include_right) for end in ends])
            
    
    def all_commits_without_ancestors(self) -> List[str]:
        result = []
        for i in range(self.n):
            if np.sum(self.matrix[:, i]) == 0:
                result.append(self.commits[i].hexsha)
        return result
    
    def all_commits_without_children(self) -> List[str]:
        result = []
        for i in range(self.n):
            if np.sum(self.matrix[i]) == 0:
                result.append(self.commits[i].hexsha)
        return result
    
    def topo_order(self) -> Generator[str, None, None]:
        in_degree = [0] * self.n
        for i in range(self.n):
            for neighbor in self.graph[i]:
                in_degree[neighbor] += 1

        queue = deque([i for i in range(self.n) if in_degree[i] == 0])

        while queue:
            current = queue.popleft()
            yield self.commits[current].hexsha
            for neighbor in self.graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
    
    def get_commit_message(self, commit: str) -> str:
        """
        Get the commit message of a commit.
        """
        idx = self._commit_to_idx[commit]
        return self.commits[idx].message.strip()
    
    def get_full_commit(self, commit: str) -> git.Commit:
        """
        Get the full commit object of a commit.
        """
        idx = self._commit_to_idx[commit]
        return self.commits[idx]
    
    def get_patch(self, base_commit: str, next_commit: str) -> str:
        """
        Get the patch between two commits.
        """
        base_commit: git.Commit = self.commits[self._commit_to_idx[base_commit]]
        next_commit = self.commits[self._commit_to_idx[next_commit]]
        
        diff = base_commit.diff(next_commit)
        patch = set([])
        for d in diff:
            if d.change_type == 'R':
                patch.add((d.a_path, 'D'))
                patch.add((d.b_path, 'A'))
            if d.change_type == 'M':
                patch.add((d.a_path, 'M'))
            if d.change_type == 'D':
                patch.add((d.a_path, 'D'))
        # print(patch)
        # exit(0 )
        return patch
    
    def get_full_patch(self, base_commit: str, next_commit: str) -> str:
        """
        Get the patch between two commits.
        """
        base_commit: git.Commit = self.commits[self._commit_to_idx[base_commit]]
        next_commit = self.commits[self._commit_to_idx[next_commit]]
        
        diff = base_commit.diff(next_commit, create_patch=True, unified=0)
        res = ''
        # get full patch content
        for d in diff:
            res += d.diff.decode('utf-8', errors='ignore')[:2000]
        return res

    def get_version(self, commit: str) -> str:
        try:
            return self.repo.git.describe('--tags', commit)
        except:
            return None
    
    def get_main_path(self) -> List[str]:
        commit = self.commits[0]
        path = [commit.hexsha]
        while commit.parents:
            commit = commit.parents[0]
            path.append(commit.hexsha)
        return path[::-1]
    
    def _build_longest_path(self):
        """
        build the longest path in commit graph.
        """
        topo_order = [self._commit_to_idx[x] for x in self.topo_order()]
        longest_path = []
        dist = [-1] * self.n

        for node in topo_order[::-1]:
            if not self.graph[node]:
                dist[node] = 1
            for neighbor in self.graph[node]:
                dist[node] = max(dist[node], dist[neighbor] + 1)

        max_length = max(dist)
        current = dist.index(max_length)

        while current != -1:
            longest_path.append(current)
            next_node = -1
            for neighbor in self.graph[current]:
                if dist[neighbor] == dist[current] - 1:
                    next_node = neighbor
                    break
            current = next_node
        self.longest_path = [self.commits[i].hexsha for i in longest_path]
        
        self._commit_to_idx_longest_path = {commit: i for i, commit in enumerate(self.longest_path)}
        self.n_longest_path = len(self.longest_path)
        
    def get_longest_path(self) -> List[str]:
        print("first commit date:", self.commits[self._commit_to_idx[self.longest_path[0]]].authored_datetime)
        print(self.longest_path[0])
        print("last commit date:", self.commits[self._commit_to_idx[self.longest_path[-1]]].authored_datetime)
        return self.longest_path
    
    def get_next_commit(self, commit: str | None) -> str | None:
        """
        Return the next commit in the longest path after the given commit.
        If the commit is the last commit in the longest path, return None.
        If the commit is None, return the first commit in the longest path.
        """
        if commit is None:
            return self.longest_path[0]
        if commit not in self._commit_to_idx_longest_path:
            raise ValueError(f"Commit {commit} not found in the longest path.")
        if self._commit_to_idx_longest_path[commit] == self.n_longest_path - 1:
            return None
        return self.longest_path[self._commit_to_idx_longest_path[commit] + 1]

    def get_base_commit(self, commit: str | None) -> str | None:
        """
        Return the last commit in the longest path before the given commit.
        If the commit is the first commit in the longest path, return None.
        If the commit is None, return the last commit in the longest path.
        """
        if commit is None:
            return self.longest_path[-1]
        if commit not in self._commit_to_idx_longest_path:
            raise ValueError(f"Commit {commit} not found in the longest path.")
        if self._commit_to_idx_longest_path[commit] == 0:
            return None
        return self.longest_path[self._commit_to_idx_longest_path[commit] - 1]
    
    

if __name__ == "__main__":
    analyzer = CommitDAGAnalyzer(repo_name='conda')
    # test
    #print(analyzer.is_ancestor('e5550f7520d58659d725db4dd50ddd87bbcfa0fd', '116c7bea61360626cd51487594423867ac08a35b'))
    #print(analyzer.is_ancestor('116c7bea61360626cd51487594423867ac08a35b', 'e5550f7520d58659d725db4dd50ddd87bbcfa0fd'))
    # a = list(analyzer.topo_order())
    # b = list(analyzer.all_commits_without_children())
    a = list(analyzer.get_longest_path())
    print(len(a), a[0])