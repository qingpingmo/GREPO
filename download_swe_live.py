from datasets import load_dataset
import sys, subprocess, os, pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

dataset = load_dataset('SWE-bench-Live/SWE-bench-Live', split='verified')

dest = 'swe_bench_live_repos'
repos = set(dataset['repo'])
os.makedirs(dest, exist_ok=True)

def clone_repo(r):
    url = f'https://github.com/{r}.git'
    name = r.rstrip('/').split('/')[-1]
    d = os.path.join(dest, name)
    subprocess.run(['git', 'clone', url, d])
    return d


# with ThreadPoolExecutor(max_workers=8) as executor:
#     future_to_repo = {executor.submit(clone_repo, r): r for r in repos}

#     for future in as_completed(future_to_repo):
#         repo = future_to_repo[future]
#         try:
#             result = future.result()
#             print(f"Successfully cloned {repo} to {result}")
#         except Exception as e:
#             print(f"Failed to clone {repo}: {e}")
            
            
def create_repo_owner_mapping():
    """Create a mapping from repo name to owner name and save to JSON"""
    repo_owner_mapping = {}
    
    for repo in repos:
        owner_name = repo.split('/')[0]
        repo_name = repo.split('/')[1]
        repo_owner_mapping[repo_name] = owner_name
    
    # Save to JSON file
    mapping_file = os.path.join(dest, 'repo_owner_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(repo_owner_mapping, f, indent=2)
    
    print(f"Repo-owner mapping saved to {mapping_file}")
    return repo_owner_mapping

def main():
    create_repo_owner_mapping()

if __name__ == "__main__":
    main()
