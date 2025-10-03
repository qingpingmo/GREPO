import json
import re
import pandas as pd
import os

def extract_code_entities_from_rewriter(text):
  # find code_entities
  pattern_code_entities = r'\[start_of_related_code_entities\]\s*(.*?)\s*\[end_of_related_code_entities\]'
  match_code_entities = re.search(pattern_code_entities, text, re.DOTALL)
  if match_code_entities:
      code_entities = match_code_entities.group(1).strip().split('\n')
  else: # execute exceptional output
      pattern_1 = r'<start_of_related_code_entities>\s*(.*?)\s*<end_of_related_code_entities>'
      match_1 = re.search(pattern_1, text, re.DOTALL)
      pattern_2 = r'\[Start_of_Related_Code_Entities\]\s*(.*?)\s*\[End_of_Related_Code_Entities\]'
      match_2 = re.search(pattern_2, text, re.DOTALL)

      if match_1:
          code_entities = match_1.group(1).strip().split('\n')
      elif match_2:
          code_entities = match_2.group(1).strip().split('\n')
      else:
          code_entities = []
  
  # add post processing
  for idx, entity in enumerate(code_entities):
      if entity.startswith("- "):
          code_entities[idx] = entity[2:]

  return code_entities

def extract_related_keywords_from_rewriter(text):
  # find related_keywords
  pattern_related_keywords = r'\[start_of_related_keywords\]\s*(.*?)\s*\[end_of_related_keywords\]'
  match_related_keywords = re.search(pattern_related_keywords, text, re.DOTALL)
  if match_related_keywords:
      related_keywords = match_related_keywords.group(1).strip().split('\n')
  else:
      pattern_1 = r'<start_of_related_keywords>\s*(.*?)\s*<end_of_related_keywords>'
      match_1 = re.search(pattern_1, text, re.DOTALL)
      pattern_2 = r'\[Start_of_Related_Keywords\]\s*(.*?)\s*\[End_of_Related_Keywords\]'
      match_2 = re.search(pattern_2, text, re.DOTALL)

      if match_1:
          related_keywords = match_1.group(1).strip().split('\n')
      elif match_2:
          related_keywords = match_2.group(1).strip().split('\n')
      else:
          related_keywords = []
          
  # add post processing
  for idx, keyword in enumerate(related_keywords):
      if keyword.startswith("- "):
          related_keywords[idx] = keyword[2:]
  
  return related_keywords

def extract_query_from_rewriter(text):
  # match query
  pattern_query = r'\[start_of_related_queries\]\s*(.*?)\s*\[end_of_related_queries\]'
  match_query = re.search(pattern_query, text, re.DOTALL)
  if match_query:
      queries = match_query.group(1).strip().split('\n')
  else:
      pattern_1 = r'<start_of_related_queries>\s*(.*?)\s*<end_of_related_queries>'
      match_1 = re.search(pattern_1, text, re.DOTALL)
      pattern_2 = r'\[Start_of_Related_Queries\]\s*(.*?)\s*\[End_of_Related_Queries\]'
      match_2 = re.search(pattern_2, text, re.DOTALL)

      if match_1:
          queries = match_1.group(1).strip().split('\n')
      elif match_2:
          queries = match_2.group(1).strip().split('\n')
      else:
          queries = []

  # add post processing
  for idx, query in enumerate(queries):
      if query.startswith("query"):
          queries[idx] = query[9:]
      elif query.startswith("-"):
          queries[idx] = query[2:]
  queries = [query for query in queries if len(query)>0]
  return queries


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=str, default="astropy")
  args = parser.parse_args()
  test_basic_info = pd.read_json(f"Graph_Feature_Construction/{args.repo}/test_rewriter_output.json")

  # start post processing
  test_basic_info["rewriter_inferer_output"] = test_basic_info["rewriter_inferer"].apply(lambda item:extract_query_from_rewriter(item))
  test_basic_info["rewriter_extractor_output_entity"] = test_basic_info["rewriter_extractor"].apply(lambda item:extract_code_entities_from_rewriter(item))
  test_basic_info["rewriter_extractor_output_keyword"] = test_basic_info["rewriter_extractor"].apply(lambda item:extract_related_keywords_from_rewriter(item))

  rewriter_output_dict = {}
  error_case = []
  for idx, item in test_basic_info.iterrows():
    instance_id = item["pullrequest_idx"]
    entity = item["rewriter_extractor_output_entity"]
    keyword = item["rewriter_extractor_output_keyword"]
    query = item["rewriter_inferer_output"]
    # if entity or keyword or query:
    if entity and keyword and query:
        rewriter_output_dict[instance_id] = {
            "code_entity": entity,
            "keyword": keyword,
            "query": query,
            "entity": entity
        }
    else:
        error_case.append(instance_id)


    with open(f"Graph_Feature_Construction/{args.repo}/rewriter_output_post.json", 'w', encoding='utf-8') as file:
        json.dump(rewriter_output_dict, file)
  
  # save trajs
  test_basic_info.to_json(f"Graph_Feature_Construction/{args.repo}/test_rewriter_output_post.json", index=False)






