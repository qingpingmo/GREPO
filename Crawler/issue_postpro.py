from datasets import Dataset
from GithubRepoAnalysis import SPLIT_STR, issue_str_to_dict, IssueDict, issue_dict_to_str

import json
import re

from typing import TypedDict

class IssueBodyAnalysisDict(TypedDict):
    bug_desc: str  # only for bug report, the description of the bug
    expected_behavior: str  # expected behavior
    actual_behavior: str  # actual behavior
    reproduce: str  # how to reproduce the bug
    version: str  # version
    require: str  # only for feature request, the requirement
    solution: str  # how to achieve the requirement
    others: str  # other parts of the issue body, as key: value pairs, the content does not match any title pattern are also included with the title as others
    code: str  # all content contained in code blocks ```...```
    error_trace: str  # only for bug report, if there is an error, the complete error trace
    error_statement: str  # the actual error infomation, which is the last non-empty line of the error trace
    error_trace_files: str  # The file sequence in the error trace
    error_trace_files_funcs: str  # The file sequence with the line and function in the error trace
    last_error_file: str  # The last error file
    last_error_file_func: str  # The last error file with line and function
    last_error_func: str  # The last error function
    
class IssueWithBodyAnalysisDict(IssueDict):
    body_analysis: IssueBodyAnalysisDict
    

class AutoPattern:
    """
    The pattern to find the title and its content in the body of a github issue. The pattern is different for different repositories. 
    
    The pattern should have two groups, the first group is the title, and the second group is the content. 
    
    Notice that the pattern should work with `re.MULTILINE` and `re.DOTALL` flags. 
    
    If the repository is not organized in the way that the title is clear, the pattern should be None.
    """
    def __call__(self, repo_name: str) -> re.Pattern[str] | None:
        repo_name = repo_name.replace("-", "_")
        # if not hasattr(self, f"_call_{repo_name}"):
        #     raise ValueError(f"Repo {repo_name} is not supported by AutoPattern")
        return getattr(self, f"_call_default")()
    
    def set_new_pattern(self, repo_name: str, pattern: re.Pattern[str]):
        repo_name = repo_name.replace("-", "_")
        setattr(self, f"_call_default", lambda: pattern)
    
    def _call_scikit_learn(self) -> re.Pattern[str]:
        return re.compile(r'^#{2,}\s*(.+?)\s*\n(.*?)(?=^#{2,}\s|\Z)', re.MULTILINE | re.DOTALL)

    def _call_transformers(self) -> re.Pattern[str]:
        return re.compile(r'^#{2,}\s*(.+?)\s*\n(.*?)(?=^#{2,}\s|\Z)', re.MULTILINE | re.DOTALL)
    
    def _call_seaborn(self) -> None:
        return None
    
    def _call_default(self) -> re.Pattern[str]:
        return re.compile(r'^#{2,}\s*(.+?)\s*\n(.*?)(?=^#{2,}\s|\Z)', re.MULTILINE | re.DOTALL)

def body_analysis(body: str, pattern: str | re.Pattern[str] | None = None, repo_name: str | None = None) -> IssueBodyAnalysisDict:
    """
    The title_pattern should be a regex pattern that matches the title of each part of the body. For example, in sklearn, the titles are like ### xxxxx\\n.
    ------
    body: str, the body of the issue
    pattern: str, the pattern of the title of each part of the body. Now it is not used, but it is reserved for future use.
    """
    return_dict: IssueBodyAnalysisDict = {} # type: ignore
    for key in ["bug_desc", "expected_behavior", "actual_behavior", "reproduce", "version", "require", "solution", "others"]:
        return_dict[key] = ""
    if body is None:
        return return_dict
    if True:
        # first split the body given the title_pattern
        if pattern is None and repo_name is None:
            raise ValueError("Either pattern or repo_name should be given")
        elif pattern is None:
            assert repo_name is not None
            pattern = AutoPattern()(repo_name)
        elif isinstance(pattern, str):
            pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
        else:
            pattern = re.compile(pattern.pattern, re.MULTILINE | re.DOTALL)
            
    parts_with_tile = dict()
    others = body
    if pattern is None:
        pass
    else:
        assert body is not None
        matches = re.finditer(pattern, body)
        for match in matches:
            all_match, title, content = match.group(0), match.group(1), match.group(2)
            parts_with_tile[title.strip()] = content.strip()
            others = others.replace(all_match, "")
                        

    for key, value in parts_with_tile.items():
        # TODO: these function should add some test to make sure different repo can be handled correctly
        # TODO: whether there is a more general way to handle the title in different repos

        key = key.lower().strip()
        
        # if the key is in pre-defined templates
        # processed repos: sklearn, transformers, astropy, scipy, ipython, django, conda, jax
        if key in {'describe the bug', 'description', 'describe your issue.', 'what happened?'}:
            return_dict["bug_desc"] += value
            continue
        if key in {'steps/code to reproduce', 'reproduction', 'how to reproduce', 'reproducing code example'}:
            return_dict["reproduce"] += value
            continue
        if key in {'expected results', 'expected behavior'}:
            return_dict["expected_behavior"] += value
            continue
        if key in {'actual results', 'error message'}:
            return_dict["actual_behavior"] += value
            continue
        if key in {'version', 'system info', 'versions', 'scipy/numpy/python version and system information', 
                   'conda info', 'system info (python version, jaxlib version, accelerator, etc.)'}:
            return_dict["version"] += value
            continue
        if key in {'describe the workflow you want to enable', 'feature request', 'motivation', 
                   'model description', 'what is the problem this feature will solve?', 
                   'describe the desired outcome', 'is your feature request related to a problem? please describe.', 
                   'what is the idea?', 'what should happen?', 'add a description'}:
            return_dict["require"] += value
            continue
        if key in {'describe your proposed solution', 'your contribution', 
                   'provide useful links for the implementation', "describe the solution you'd like."}:
            return_dict["solution"] += value
            continue
        
        
        # else, fuzzy matching
        if "bug" in key or (("describe" in key or "description" in key) and "feature" not in key):
            return_dict["bug_desc"] = value
        elif "expected" in key:
            return_dict["expected_behavior"] = value
        elif "actual" in key or "error" in key:
            return_dict["actual_behavior"] = value
        elif "reproduce" in key or "reproducing" in key:
            return_dict["reproduce"] = value
        elif "version" in key:
            return_dict["version"] = value
        elif "want" in key or "feature" in key:
            return_dict["require"] = value
        elif "solution" in key:
            return_dict["solution"] = value
        else:
            return_dict["others"] += f"{key}: {value}\n\n"
            
    # If anyother parts are not below a title, we also add them to others with the title as others
    if others.strip():
        return_dict["others"] += f"others: {others.strip()}\n\n"

    # extract code
    code_pattern = r'```(.+?)```'
    code_matches = re.findall(code_pattern, body, re.MULTILINE | re.DOTALL)
    return_dict["code"] = "\n####\n\n".join(code_matches)
    
    # extract error trace, find "Traceback (most recent call last):" in each code block until the end of the code block"
    error_trace_pattern = r'Traceback \(most recent call last\):(.+?)(?=^```|\Z)'
    error_trace_matches = re.findall(error_trace_pattern, body, re.MULTILINE | re.DOTALL)
    return_dict["error_trace"] = "\n####\n\n".join(error_trace_matches)
    
    # extract error statement, we assume the last non-empty line of the error trace is the error statement
    error_statement_pattern = r'(?<=\n)([^\n]+)\n*$'
    error_statement_matches = []
    for etm in error_trace_matches:
        error_statement = re.findall(error_statement_pattern, etm, re.MULTILINE)
        if error_statement:
            error_statement_matches.append(error_statement[-1])
    return_dict["error_statement"] = "\n####\n\n".join(error_statement_matches)
    
    # extract error trace files, like File "xxx/xxx/xxx/xxxx.yy" line lll, in funcname
    error_trace_files_pattern = r'File "(.+?)", line (\d+), in (.+?)(?:\s*\n|$)'
    error_trace_files_matches = re.findall(error_trace_files_pattern, body, re.MULTILINE|re.DOTALL)
    return_dict["error_trace_files"] = ", ".join([file for file, _, _ in error_trace_files_matches])
    return_dict["error_trace_files_funcs"] = "\n".join([f"File {file}, line {line}, in {func}" for file, line, func in error_trace_files_matches])
    
    # extract the last error file, func, and line
    if error_trace_files_matches:
        return_dict["last_error_file"], _, return_dict["last_error_func"] = error_trace_files_matches[-1]
        return_dict["last_error_file_func"] = "File {}, line {}, in {}".format(*error_trace_files_matches[-1])
    else:
        return_dict["last_error_file"], return_dict["last_error_file_func"], return_dict["last_error_func"] = "", "", ""
    
    print(f"Body analysis for repo {repo_name}:\n{json.dumps(return_dict, indent=4)}", flush=True)
    return return_dict    

def issue_postprocess(repo_name: str):
    pulldata = Dataset.load_from_disk(f"pulldata/{repo_name}")
    
    return_list = []
    for pr in pulldata:
        issue_numbers = pr["issues"][1:] # The first issue is a dummy -1 # type: ignore
        # due to some bugs, there could be repeated issues, we need to remove them
        issue_numbers_no_repeat = []
        index = []
        for i, issue in enumerate(issue_numbers):
            if issue not in issue_numbers_no_repeat:
                index.append(i)
                issue_numbers_no_repeat.append(issue)
        issue_numbers = issue_numbers_no_repeat
        try:
            issue_info = issue_str_to_dict(pr["issues_info"]) # type: ignore
        except json.JSONDecodeError:
            pr["raw"] = pr["issues_info"]
            
        issue_info = [issue_info[i] for i in index] # remove the repeated issues
        assert len(issue_info) == len(issue_numbers)
        # for issue in issue_info:
        #     print(issue['body'])
        #     print('========================\n')
        #     print(issue['labels'])
        #     print('------------------------')
        # input()
        body_analysis_list = [body_analysis(issue["body"], repo_name=repo_name) for issue in issue_info]
        issue_ana: list[IssueWithBodyAnalysisDict] = []
        for idx in range(len(issue_numbers)):
            issue_ana.append({**issue_info[idx], "body_analysis": body_analysis_list[idx]})
        pr["issues_info"] = issue_dict_to_str(issue_ana) # type: ignore
        pr["issues"] = [-1] + issue_numbers
        return_list.append(pr)
    return return_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_name", type=str, help="The name of the repository")
    args = parser.parse_args()
    repo_name = args.repo_name
    pulldata = Dataset.load_from_disk(f"pulldata/{repo_name}")
    
    print(f"Loaded {len(pulldata)} pull requests from {repo_name}")
    new_pulldataset = Dataset.from_list(issue_postprocess(repo_name))
    # print(json.dumps(json.loads(new_pulldataset[1]['issues_info']), indent=4))
    new_pulldataset.save_to_disk(f"pulldata/{repo_name}_postpro")