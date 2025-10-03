"""
Generate Prompts for Rewriter
"""

import pandas as pd

def generate_prompt_for_extractor_v1(problem_statement, repo_name):
    """
    generate prompt for extractor
    """
    
    prompt = """
<issue>
{}
</issue> 
This is an issue related to repository '{}'. 
Instructions:
1. Analysis:
  ○ Analyze the provided issue description. Identify the relevant File, Class, or Function involved.
  ○ Determine the specific problem or error encountered and note any clues that may assist in locating the relevant or problematic area.
2. Extraction:
  ○ After the analysis, extract ALL the mentioned code entities (File, Class, or Function), especially Files.
  ○ Then extract three potential and meaningful keywords, responding in the following format:

[start_of_analysis] 
<detailed_analysis> 
[end_of_analysis] 

[start_of_related_code_entities] 
<entity_name_with_path>
[end_of_related_code_entities] 

[start_of_related_keywords] 
<keywords>
[end_of_related_keywords]

Notes:
- Pay attention to the information in the error logs (if exists).
- The buggy code exists solely in the project described in the issue (e.g., django, sklearn). Buggy location is usually not in the tests files or external packages.
- Your extracted entities should be CONCISE, ACCURATE and INFORMATIVE.
- Provide the relative path for code entities if specified (e.g., package/foo.py). Relative path is relative to the repository itself, do not include suffix like '/home/username/', '/etc/service/' or '/tree/master'.
- Do not include any additional information such as line numbers or explanations in your extraction result.

Preferred extraction Examples of Code Entities:
- repo/cart.py
- Class User()
- def getData()
Preferred extraction Examples of Keywords:
- train_loop
- hooks
- docker
  
Unpreferred extraction Examples of keywords:
- something wrong
- input validation
- TypeError
    """.format(problem_statement, repo_name)
    
    return prompt

def generate_prompt_for_inferer_v1(problem_statement, repo_name):
    """
    generate prompt for inferer
    """
    
    prompt = """
<issue>
{}
</issue> 
This is an issue related to repository '{}'. 
Task:
Based on the issue description provided, identify the characteristics of code entities (files, functions, class) that might need to be modified. 
For each characteristic, generate a search query that could help locate relevant code entities in a codebase.
Instructions:
First, analyze the issue description and identify keywords, features, and functionalities that are likely relevant to the modification of code entities.
Then, create queries that capture these characteristics, focusing on:
● File names that may implement relevant functionalities.
● Functions or methods that are related to the features described in the issue.
● Any patterns or structures that might be relevant to the functionalities mentioned.
For example:
● File related to the initialization of a neural network.
● Function related to the training process.
● Code used to configure the service.
Please answer in the following format:

[start_of_analysis] 
<detailed_analysis> 
[end_of_analysis] 

[start_of_related_queries] 
query 1:
query 2:
...
[end_of_related_queries] 

Notes:
- Your queries should be DETAILED, ACCURATE and INFORMATIVE. 
- Your queries should be a complete sentences and do not include additional explanation.
- The number of queries is up to five, so be focus on the important characteristics.
- Your queries should focus on the repository code itself, rather than other information like commit history.
- Pay attention to the information in the error logs (if exists).

Preferred Query Examples:
- Look for references to "tqdm" or "progress_bar" within the training loop files to find where progress bars are currently updated.
- Code snippets where 'gethostbyname' function from 'socket' module is called.
- File name containing 'mysql.py' AND functions related to 'MySQLStatementSamples' initialization.
- Functions or methods handling hostname resolution or encoding within 'datadog_checks' directory.
- Find all occurrences of "early_stopping" within files that also mention "Trainer" to identify where early stopping logic is implemented and potentially needs adjustment for non-default 'val_check_interval'.
    """.format(problem_statement, repo_name)
    
    return prompt

if __name__ == "__main__":

  from datasets import Dataset
  import json
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=str, default="astropy")
  args = parser.parse_args()
  #reponame = "astropy"
  reponame = args.repo
  dataset = Dataset.load_from_disk(f"pulldata/{reponame}_postpro/")
  extractor_prompt_list = []
  inferer_prompt_list = []
  idxlist = []
  for idx, item in enumerate(dataset):
    if item["issues_info"]:
      for line in item["issues_info"].split("#@!@#"):
        issues_info = json.loads(line)# ['number', 'title', 'body', 'labels', 'comments', 'body_analysis']
        title = issues_info["title"] or ""
        body = issues_info["body"] or ""
        tmp_extractor_prompt = generate_prompt_for_extractor_v1(title+"\n"+body, reponame)
        tmp_inferer_prompt = generate_prompt_for_inferer_v1(title+"\n"+body, reponame)
        extractor_prompt_list.append(tmp_extractor_prompt)
        inferer_prompt_list.append(tmp_inferer_prompt)
        idxlist.append(idx)
        
  test_basic_info = pd.DataFrame.from_dict({
    "extractor_prompt": extractor_prompt_list,
    "inferer_prompt": inferer_prompt_list,
    "pullrequest_idx": idxlist
  })
  import os
  if not os.path.exists(f"Graph_Feature_Construction/{args.repo}/"):
      os.makedirs(f"Graph_Feature_Construction/{args.repo}/")
  
  test_basic_info.to_json(f"Graph_Feature_Construction/{args.repo}/test_rewriter_prompt.json", index=False)

