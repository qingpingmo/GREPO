import os
import sys
import time
from tqdm import tqdm
import re
import json
from copy import deepcopy
import argparse
import logging
import pandas as pd
from openai import OpenAI
import random
from typing import List, Optional


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIInference:
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-flash", max_retries: int = 5):

        #self.api_key = "sk-xxx" #api_key or os.getenv("OPENAI_API_KEY")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set and no api_key provided")
        
        self.client = OpenAI(api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = model
        self.max_retries = max_retries
    
    def _use_chat_endpoint(self) -> bool:
        
        m = self.model.lower()
        
        if m.startswith("o1") or m.startswith("o3"):
            return False
        if "o3" in m or "o1" in m:
            return False
        return True  
    
    def generate_response(self, prompt: str, retry_count: int = 0) -> str:

        try:
            if self._use_chat_endpoint():
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # max_tokens=2048,
                    # temperature=0.7,
                    # top_p=0.9
                )
                return (resp.choices[0].message.content or "").strip()
            else:
                
                full_input = (
                    "System: You are a helpful assistant.\n" +
                    "User: " + prompt
                )
                # responses.create 
                resp = self.client.responses.create(
                    model=self.model,
                    input=full_input,
                    # max_output_tokens=2048,
                    # temperature=0.7,
                )
                
                text = getattr(resp, "output_text", None)
                if not text:
                    
                    parts = []
                    for out in getattr(resp, "output", []) or []:
                        if out.get("type") == "output_text":
                            parts.append(out["text"]["content"])
                    text = "".join(parts)
                return (text or "").strip() or "error"
        except Exception as e:
            logger.warning(f"API call failed ( Attempt {retry_count + 1}/{self.max_retries}): {e}")
            if retry_count < self.max_retries - 1:
                
                sleep_time = min(60, (2 ** retry_count)) + random.uniform(0, 1)
                logger.info(f"Retrying after waiting for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
                return self.generate_response(prompt, retry_count + 1)
            logger.error("Reached the maximum number of retries, returning an error flag")
            return "error"
    
    def generate_batch_responses(self, prompts: List[str], batch_size: int = 8) -> List[str]:

        responses = []
        logger.info(f"Starting to process {len(prompts)} prompts... (batch_size={batch_size})")
        
        responses = [self.generate_response(prompt) for prompt in prompts]#  tqdm(prompts)]
        
        #from pqdm.processes import pqdm
        #responses = pqdm(prompts, self.generate_response, n_jobs=32) 

        '''
        for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
            batch_prompts = prompts[i:i + batch_size]
            batch_responses = []
            for prompt in batch_prompts:
                batch_responses.append(self.generate_response(prompt))
                time.sleep(0.05)  
            responses.extend(batch_responses)
            if i + batch_size < len(prompts):
                time.sleep(0.3)
        '''
        return responses

###################################################

def parse_args():

    
    parser = argparse.ArgumentParser(description="Run Inference Model with OpenAI API.")
    
    parser.add_argument('--prompt_path', nargs='?', default='',
                        help='Specify the prompts file')
    
    parser.add_argument('--model', nargs='?', default='qwen-flash',
                        help='OpenAI model to use (default: o3)')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing (default: 8)')
    
    parser.add_argument('--max_retries', type=int, default=5,
                        help='Maximum number of retries for API calls (default: 5)')
    
    parser.add_argument("--repo", type=str, default="astropy")

    return parser.parse_args()


def infer_subroutine(data, batch_size, model, max_retries):
    
    try:
        inference_engine = OpenAIInference(model=model, max_retries=max_retries)
        logger.info(f"Successfully initialized OpenAI client, using model: {model}")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return False

    
    try:
        response_list = inference_engine.generate_batch_responses(data, batch_size=batch_size)
        logger.info(f"Successfully generated {len(response_list)} responses")
    except Exception as e:
        logger.error(f"Failed to generate response: {e}")
        return False
    return response_list

def inference_LLM_patch(prompt_path: str, model: str = "o3", batch_size: int = 8, max_retries: int = 5):

    try:
        test_basic_info = pd.read_json(prompt_path)
        logger.info(f"Successfully loaded data: {prompt_path}")
    except Exception as e:
        logger.error(f"Failed to loaded data: {e}")
        return False
    
    
    extractor_prompts = test_basic_info["extractor_prompt"].tolist()
    inferer_prompts = test_basic_info["inferer_prompt"].tolist()
    data = extractor_prompts + inferer_prompts
    instance_num = len(extractor_prompts)
    

    logger.info(f"Ready to process {instance_num} instances, with {len(data)} prompts")
    
    from pqdm.threads import pqdm
    import functools

    batchs = [data[i: i+batch_size] for i in range(0, len(data), batch_size)]

    tinfer_subroutine = functools.partial(infer_subroutine, batch_size=batch_size, model=model, max_retries=max_retries)

    response_list = pqdm(batchs, tinfer_subroutine, n_jobs=64)
    response_list = sum(response_list, start=[])
    
    
    if len(response_list) != len(data):
        logger.error(f"#reponses mismatch: expecting {len(data)}, actually got {len(response_list)}")
        return False
    
    
    test_basic_info["rewriter_extractor"] = response_list[:instance_num]
    test_basic_info["rewriter_inferer"] = response_list[instance_num:]

    import os
    if not os.path.exists(f"Graph_Feature_Construction/{args.repo}/"):
        os.makedirs(f"Graph_Feature_Construction/{args.repo}/")
    
    output_file = f"Graph_Feature_Construction/{args.repo}/test_rewriter_output.json"
    try:
        test_basic_info.to_json(output_file, orient='records', indent=2)
        logger.info(f"Save results to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return False
    
   
    error_count = sum(1 for resp in response_list if resp == "error")
    success_count = len(response_list) - error_count
    
    logger.info(f"Finished - SUC: {success_count}, FAI: {error_count}")
    
    return True

if __name__ == "__main__":
    
    args = parse_args()
    
    if not args.prompt_path:
        logger.error("--prompt_path lost")
        sys.exit(1)
    
    logger.info("Starting...")
    
    success = inference_LLM_patch(
        prompt_path=args.prompt_path,
        model=args.model,
        batch_size=args.batch_size,
        max_retries=args.max_retries
    )
    
    if success:
        logger.info("Finished.")
    else:
        logger.error("FAILED.")
        sys.exit(1)