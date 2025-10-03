import os
import json
from openai import OpenAI
from commit_utils import CommitDAGAnalyzer
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, TypedDict, Optional
from pydantic import BaseModel as BaseModel_query
import concurrent.futures
import random 
from tqdm import tqdm
import argparse
from tooldantic import OpenAiResponseFormatBaseModel as BaseModel
import datasets
from tasks.utils import parse_evaluation_json, LLMClientManager, BatchManagerOpenAI

class ClassifyResponse(BaseModel):
    thoughts: str
    category: List[int]

class ClassifyResponseQuery(BaseModel_query):
    thoughts: str
    category: List[int]

class PartitionResponseQuery(BaseModel_query):
    IsBug: bool
    Bug_description: str
    Reproduction: str
    Expected_Behavior: str
    Actual_Behavior: str
    Environment: str
    Other: str



QUERY_COMMIT = '''
Below is a commit message extracted from a commit from a repository. Your task is to classify the commit into one of the following categories:
1. **Bug Fix**: The commit addresses a bug or issue in the code.
2. **Feature Addition**: The commit introduces a new feature or functionality.
3. **Documentation Update**: The commit updates or adds documentation.
4. **Testing**: The commit adds or modifies tests.
5. **Refactoring**: The commit improves the code structure, organization, formatting, or style without changing its functionality (**includes removal, code cleanup, or infrastructure updates**).
6. **Other**: The commit does not fit any of the above categories.

If the commit fit multiple (>0) categories, please provide a list of all applicable categories in the "category" field.

You should respond with a JSON object with the following format:
```json
{{
    "thoughts": str,  # Your reasoning about the commit classification
    "category": List[int],  # List of categories, e.g., [1, 2] for Bug Fix and Feature Addition
}}
```

Please classify the commit below:
<commit>
{commit_message}
</commit>


Note that your response should be a valid JSON object.
'''

QUERY_ISSUE = '''
Here is a pull request along with the related issues it addresses, sourced from a GitHub repository. Your task is to classify the PR into one of the following categories:
1. **Bug Fix**: The PR addresses a bug or issue in the code.
2. **Feature Addition**: The PR introduces a new feature or functionality.
3. **Documentation Update**: The PR updates or adds documentation.
4. **Testing**: The PR adds or modifies tests.
5. **Refactoring**: The PR improves the code structure, organization, formatting, or style without changing its functionality (**includes removal, code cleanup, or infrastructure updates**).
6. **Other**: The PR does not fit any of the above categories.

If the PR fit multiple (>0) categories, please provide a list of all applicable categories in the "category" field.

You should respond with a JSON object with the following format:
```json
{{
    "thoughts": str,  # Your reasoning about the PR classification
    "category": List[int],  # List of categories, e.g., [1, 2] for Bug Fix and Feature Addition
}}
```

Please classify the PR below:
<pr>
{pr_str}
</pr>

The issues related to this PR are as follows:
<issues>
{issue_str}
</issues>
<issue_labels>
{issue_labels}
</issue_labels>

Note that your response should be a valid JSON object.
'''

QUERY_ISSUE_PARTITION_BUG = '''
Your task is to categorize the content of an issue into the following sections **if it is a bug report**. Use the exact original text from the issue—do not modify or paraphrase anything. Assign each part to **exactly one** category. If a category has no relevant content, leave it empty. Return a json format dictionary with the specified keys:  

1. Bug Description: A brief description of the bug.  
2. Reproduction: Steps or code to reproduce the bug.  
3. Expected Behavior: What should happen under normal conditions.  
4. Actual Behavior: What actually occurs due to the bug.  
5. Environment: The environment where the bug appears (e.g., OS, version).  
6. Other: Any content that doesn’t fit the above categories.  

**Rules**:  
- FIRST CHECK WHETHER THE ISSUE IS A BUG REPORT. If the issue is not a bug report, set `IsBug` to False and return an empty string for other fields.
- Preserve the original wording.  
- Do not omit or merge sentences.  
- If a category is empty, set its value to an empty string ("").  

The issue content is as follows:
<issues>
{issue_str}
</issues>

Please respond with a JSON object in the following format:
```json
{{
    "Thoughts": "Your reasoning about the issue categorization",
    "IsBug": True or False,
    "Bug Description": "[original text]",  
    "Reproduction": "[original text]",  
    "Expected Behavior": "[original text]",  
    "Actual Behavior": "[original text]",  
    "Environment": "[original text]",  
    "Other": "[original text]"  
}}
```  
'''


QUERY_ISSUE_PARTITION_FEATURE = '''
Your task is to categorize the content of an issue into the following sections **if it is a feature request**. Use the exact original text from the issue—do not modify or paraphrase anything. Assign each part to **exactly one** category. If a category has no relevant content, leave it empty. Return a json format dictionary with the specified keys:  

1. Feature Description: Description of the feature.  
2. Proposed Solution: Suggested implementation or solution for the feature.
3. Other: Any content that doesn’t fit the above categories.

**Rules**:  
- FIRST CHECK WHETHER THE ISSUE IS A FEATURE REQUEST. If the issue is not a feature request, set `IsFeature` to False and return an empty string for other fields.
- Preserve the original wording.  
- Do not omit or merge sentences.  
- If a category is empty, set its value to an empty string ("").  

The issue content is as follows:
<issues>
{issue_str}
</issues>

Please respond with a JSON object in the following format:
```json
{{
    "Thoughts": "Your reasoning about the issue categorization",
    "IsFeature": True or False,
    "Feature Description": "[original text]",
    "Proposed Solution": "[original text]"
    "Other": "[original text]"
}}
```  
'''

llmclient = LLMClientManager()

def get_commits_queries(repo, sample_num):
    analyzer = CommitDAGAnalyzer(repo_name=repo)
    longest_path = analyzer.get_longest_path()
    num = len(longest_path)
    print(f"repo: {repo}, num: {num}")
    kk_dict = {}
    query_list = []
    random.seed(42)
    if sample_num != 0:
        idx = random.sample(range(1, num - 1), sample_num)
    else:
        idx = range(1, num - 1)
    for i in idx:
        prev_commit = longest_path[i - 1]
        base_commit = longest_path[i]
        # patch = analyzer.get_full_patch(prev_commit, base_commit)
        patch = ''
        commit_message = analyzer.get_commit_message(base_commit)
        query_list.append({
            'custom_id': f"{repo}_{base_commit}",
            'content': QUERY_COMMIT.format(commit_message=commit_message, patch=patch)
        })
        kk_dict[QUERY_COMMIT.format(commit_message=commit_message, patch=patch)] = f"{repo}_{base_commit}"
    
    return query_list, kk_dict

def dict2str(d, keys):
    return '\n'.join([f"{key.upper()}:\n{d[key]}\n" for key in keys if key in d and d[key] is not None]).strip()

def get_prs_queries(repo, sample_num):
    prs = datasets.Dataset.load_from_disk(f'/home/haotong/openhands/RepoGNN/new_issue_data/{repo}')
    random.seed(42)
    # sample issues
    if sample_num != 0:
        prs = prs.select(range(sample_num))
    
    # features: ['number', 'title', 'state', 'body', 'base_commit_sha', 'timestamp', 'files', 'file_patches', 'commit', 'review_comment', 'comment', 'review', 'issues', 'issues_info', 'participants'],
    query_list = []
    for pr in prs:
        pr_str = dict2str(pr, ['title', 'body', 'comment'])
        if pr['issues_info'] is None or pr['issues_info'] == '':
            issues = []
        else:
            issues = [json.loads(i) for i in pr['issues_info'].split('#@!@#')]
        print(pr['issues'])
        for issue in issues:
            issue_str = dict2str(issue, ['title', 'body'])
            query_list.append({
                'custom_id': f"{repo}_{issue['number']}",
                'content': QUERY_ISSUE_PARTITION_BUG.format(issue_str=issue_str),
            })
    # with open('test.json', 'w') as f:
    #     json.dump(query_list, f, indent=4)
    return query_list

def classify(repo, sample_num):
    analyzer = CommitDAGAnalyzer(repo_name=repo)
    longest_path = analyzer.get_longest_path()
    query_list, kk_dict = get_commits_queries(repo, sample_num)
    query_list = [(q['content'], '') for q in query_list]  # empty system message
    res_list = llmclient.get_responses_in_parallel(query_list)
    print("cost:", llmclient.get_cost())
    print(len(res_list))
    new_dataset = []
    for i, (msg, res) in enumerate(res_list):
        for _ in range(50):
            try:
                res = parse_evaluation_json(res)
                break
            except Exception as e:
                print("################")
                (msg, res), _ = llmclient.get_response(msg, '')
        # res['commit_message'] = analyzer.get_commit_message(longest_path[i])
        res['commit_message'] = msg
        res['id'] = kk_dict[msg]
        # res['commit_id'] = longest_path[i]
        new_dataset.append(res)
    
    with open('./commit_{}_dpsk.json'.format(repo), 'w') as f:
        json.dump(new_dataset, f, indent=4)
        

def conver2pandas():
    with open('issue_classify_result.json', 'r') as f:
        data = json.load(f)
    new_data = []
    for item in data:
        tmp = {
            'repo': item['custom_id'].split('_')[0],
            'number': item['custom_id'].split('_')[1],
            'category': item['response']['category'],
        }
        new_data.append(tmp)
    df = datasets.Dataset.from_list(new_data)
    df.save_to_disk('issue_classify_result_dataset')

def partition_issues():
    query_dict_list = []
    for repo in ['astropy']:
        query_dict_list.extend(get_prs_queries(repo, args.num))
    query_list = [(q['content'], '') for q in query_dict_list]  # empty system message
    res_list = llmclient.get_responses_in_parallel(query_list)
    result = []
    for id, (msg, res) in enumerate(res_list):
        result.append({
            'custom_id': query_dict_list[id]['custom_id'],
            'instruction': msg,
            'response': res
        })
        # print(res)
        # exit(0)
    with open('issue_partition_bug_result.json', 'w') as f:
        json.dump(result, f, indent=4)

def merge_partitions():
    with open('/home/haotong/openhands/RepoGNN/issue_partition_bug_result.json', 'r') as f:
        bug_data = json.load(f)
    with open('/home/haotong/openhands/RepoGNN/issue_partition_feature_result.json', 'r') as f:
        feature_data = json.load(f)
    new_data = {}
    for item in bug_data:
        d = parse_evaluation_json(item['response'][1])
        if item['custom_id'] not in new_data and d['IsBug']:
            try:
                new_data[item['custom_id']] = {
                    'category': 1,  # Bug Fix
                    'bug_description': d['Bug Description'] if 'Bug Description' in d else d['Bug description'],
                    'reproduction': d['Reproduction'],
                    'expected_behavior': d['Expected Behavior'],
                    'actual_behavior': d['Actual Behavior'],
                    'environment': d['Environment'],
                    'other': d['Other'],
                }
            except KeyError as e:
                print(e, d.keys())
    for item in feature_data:
        d = parse_evaluation_json(item['response'][1])
        if item['custom_id'] in new_data and d['IsFeature']:
            print("both bug and feature", item['custom_id'])
            continue
        if item['custom_id'] not in new_data and d['IsFeature']:
            new_data[item['custom_id']] = {
                'category': 2,  # Feature Addition
                'feature_description': d['Feature Description'],
                'proposed_solution': d['Proposed Solution'],
                'other': d['Other'],
            }
    print(len(new_data))
    with open('issue_partition_result.json', 'w') as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['gpt-4o-mini', 'deepseek-chat', 'o4-mini', 'gpt-4.1-mini', 'deepseek-reasoner'],
    )
    parser.add_argument(
        '--num',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Use batch mode for OpenAI API',
    )
    parser.add_argument(
        '--issues',
        action='store_true',
        help='Get issues from the dataset',
    )
    args = parser.parse_args()
    llmclient.switch_model(args.model)
    # partition_issues()
    merge_partitions()
    exit(0)
    tmp = datasets.Dataset.load_from_disk('/home/haotong/openhands/RepoGNN/new_issue_data/astropy').to_pandas()
    print(tmp.iloc[0]['body'])
    # print(tmp[0])
    # exit(0)
    # conver2pandas()
    exit(0)
    if args.issues:
        query_list = []
        for repo in ['scikit-learn', 'conda', 'ipython', 'jax', 'scipy', 'transformers','django', 'astropy' ]:
            query_list.extend(get_prs_queries(repo, args.num))
        print(len(query_list))
        result = []
        for query in query_list:
            res, _ = llmclient.get_response(query['content'], '', response_format=ClassifyResponseQuery)
            result.append({
                'custom_id': query['custom_id'],
                'response': res
            })
        with open('issue_classify_result.json', 'w') as f:
            json.dump(result, f, indent=4)
        
            
        exit(0)
    if args.batch:
        
        all_repo = []
        # for repo in ['scikit-learn', 'conda', 'ipython', 'jax', 'scipy', 'transformers','django', 'astropy' ]:
        #     query_list, idx = get_commits_queries(repo, args.num)
        #     all_repo.extend(query_list)
        all_repo = [1] * 190000
        print("#")
        results = []
        for i in range(0, len(all_repo), 50000):
            tmp = all_repo[i:i + 50000]
            batch_manager = BatchManagerOpenAI(f'3classify_commits_{i}', args.model)
            # batch_manager.get_status()
            # batch_manager.create_jsonl_file(tmp, response_format=ClassifyResponse.model_json_schema())
            # batch_manager.upload_and_submit()
            batch_manager.get_status()
            # conver2pandas()
            if os.path.exists(batch_manager.result_file):
                with open(batch_manager.result_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    if 'response' in data:
                        res = json.loads(data['response']['body']['choices'][0]['message']['content'])
                        if data['custom_id'] is None:
                            print(data)
                            print(i)
                            exit(0)
                        results.append({
                            'repo': data['custom_id'].split('_')[0],
                            'commit_id': data['custom_id'].split('_')[1],
                            'category': res['category'],
                        })
        # to pandas
        df = datasets.Dataset.from_list(results)
        df.save_to_disk('commit_classify_result_dataset')
        exit(0)
    else:
        kkk = 0
        llmclient.switch_model(args.model)
        for repo in ['scikit-learn', 'conda', 'ipython', 'jax', 'scipy', 'transformers','django', 'astropy' ]:
            classify(repo, args.num)