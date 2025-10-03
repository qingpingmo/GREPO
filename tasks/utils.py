import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm import tqdm
from openai import OpenAI

def parse_evaluation_json(evaluation_text):
    if not isinstance(evaluation_text, str):
        print(f"Warning: Expected string for parsing, got {type(evaluation_text)}. Content: {evaluation_text}")
        return None
    evaluation_text = evaluation_text.split('JSON:')[-1]
    try:
        return json.loads(evaluation_text)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*\n(.*?)\n```', evaluation_text, re.DOTALL | re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse extracted JSON: {e}\nContent: {json_str}")
                return None
        else:
            print(f"Warning: Could not find JSON block or parse the string directly:\n{evaluation_text}...")
            


class LLMClientManager:
    MAX_NUM_TOKENS = 8162
    total_cost = 0

    def __init__(self):
        self.openai_client = None
        self.deepseek_client = None
        self.current_client = None
        self.current_model = None
        self.msg_history = []

    def switch_model(self, model: str):
        print('Switching model to:', model)
        self.current_model = model
        if model in ["gpt-4o", "gpt-4o-mini", "o3-mini", "o4-mini", 'gpt-4.1-mini']:
            if self.openai_client is None:
                API_KEY = os.getenv("OPENAI_API_KEY")
                self.openai_client = OpenAI(api_key=API_KEY)
            self.current_client = self.openai_client
            
        elif model in ["deepseek-chat", "deepseek-reasoner", 'Pro/deepseek-ai/DeepSeek-R1', 'deepseek-r1-250120']:
            if self.deepseek_client is None:
                API_KEY = os.getenv("DEEPSEEK_API_KEY")
                self.deepseek_client = OpenAI(api_key=API_KEY, base_url='https://api.deepseek.com')
            self.current_client = self.deepseek_client
        else:
            raise ValueError(f"Model {model} not supported.")

    def get_response(self, msg, system_message, response_format=None, temperature=0.3, print_debug=False):
        if self.current_client is None or self.current_model is None:
            self.switch_model("gpt-4o-mini")
        
        msg_history = self.msg_history
        for _ in range(3):
            try:
                if self.current_model in ["o3-mini", "o4-mini", "gpt-4o", "gpt-4o-mini"]:
                    new_msg_history = msg_history + [{"role": "user", "content": msg}]
                    if response_format is not None:
                        response = self.current_client.beta.chat.completions.parse(
                            model=self.current_model,
                            messages=[{"role": "user", "content": system_message}, *new_msg_history],
                            temperature=temperature,
                            max_completion_tokens=self.MAX_NUM_TOKENS,
                            n=1,
                            response_format=response_format
                        )
                    else:
                        response = self.current_client.chat.completions.create(
                            model=self.current_model,
                            messages=[{"role": "system", "content": system_message}, *new_msg_history],
                            temperature=temperature,
                            max_completion_tokens=self.MAX_NUM_TOKENS,
                        )
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    if self.current_model in ['o3-mini', 'o4-mini']:
                        self.total_cost += completion_tokens * 4.4 / 1000000 + prompt_tokens * 1.1 / 1000000
                    elif self.current_model in ['gpt-4o-mini']:
                        self.total_cost += completion_tokens * 0.6 / 1000000 + prompt_tokens * 0.15 / 1000000
                    elif self.current_model in ['gpt-4o']:
                        self.total_cost += completion_tokens * 10 / 1000000 + prompt_tokens * 0.5 / 1000000
                    content = response.choices[0].message.content
                    if response_format is not None:
                        content = json.loads(content)
                    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
                
                elif self.current_model in ["deepseek-chat"]:
                    new_msg_history = msg_history + [{"role": "user", "content": msg}]
                    response = self.current_client.chat.completions.create(
                        model=self.current_model,
                        messages=[{"role": "system", "content": system_message}, *new_msg_history],
                        temperature=temperature,
                        max_tokens=self.MAX_NUM_TOKENS,
                        n=1,
                        stop=None,
                    )
                    content = response.choices[0].message.content
                    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
                
                elif self.current_model in ["deepseek-reasoner", 'Pro/deepseek-ai/DeepSeek-R1', 'deepseek-r1-250120']:
                    new_msg_history = msg_history + [{"role": "user", "content": msg}]
                    response = self.current_client.chat.completions.create(
                        model=self.current_model,
                        messages=[{"role": "system", "content": system_message}, *new_msg_history],
                        n=1,
                        stop=None,
                        timeout=120
                    )
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    self.total_cost += completion_tokens * 2.19 / 1000000 + prompt_tokens * 0.55 / 1000000
                    content = (response.choices[0].message.reasoning_content, response.choices[0].message.content)
                    new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
                
                else:
                    raise ValueError(f"Model {self.current_model} not supported.")
                
                break
            except Exception as e:
                print("Retrying...")
                print(e)
                continue

        
        # self.msg_history = new_msg_history
        return (msg, content), new_msg_history

    def clear_cost(self):
        self.total_cost = 0

    def get_cost(self):
        return self.total_cost

    def get_responses_in_parallel(self, prompt_system_pairs: list):
        with ThreadPoolExecutor(max_workers=64) as executor:
            results = list(tqdm(
                executor.map(lambda pair: self.get_response(pair[0], pair[1]), prompt_system_pairs),
                total=len(prompt_system_pairs),
                desc="Processing"
            ))
        responses = []
        for result in results:
            try:
                (msg, response), _ = result
                responses.append((msg, response))
            except Exception as e:
                print(f"Error processing a request: {e}")
                responses.append(None)
        return responses
    
class BatchManagerOpenAI:
    def __init__(self, exp_name, model):
        exp_name += '_' + model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.folder = './batch/' + exp_name + '/'
        self.query_file = self.folder + 'query.jsonl'
        self.result_file = self.folder + 'result.jsonl'
        self.name = exp_name
        self.model = model
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    
    def create_jsonl_file(self, data, system='', response_format=None):
        query_list = []
        token_string = 'max_tokens'
        if self.model in ['o4-mini']:
            token_string = 'max_completion_tokens'
        for t in data:
            query_list.append({
                'custom_id': t['custom_id'],
                'method': 'POST',
                'url': '/v1/chat/completions',
                'body': {
                    'model': self.model,
                    'messages': [
                        {
                            'role': 'system',
                            'content': system
                        },
                        {
                            'role': 'user',
                            'content': t['content']
                        }
                    ],
                    token_string: 8192,
                    "response_format": response_format
                }
            })

        with open(self.query_file, 'w') as file:
            for query in query_list:
                file.write(json.dumps(query) + '\n')

    def upload_and_submit(self):
        batch_input_file = self.client.files.create(
            file=open(self.query_file, "rb"),
            purpose="batch"
        )
        
        batch_input_file_id = batch_input_file.id
        tmp = self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": self.name
            }
        )
        print(tmp)
        self.batch_id = tmp.id
        with open(self.folder + 'batch_id.txt', 'w') as f:
            f.write(tmp.id)

    def get_status(self):
        with open(self.folder + 'batch_id.txt', 'r') as f:
            batch_id = f.read()
        batch = self.client.batches.retrieve(batch_id)
        print(batch)
        if batch.output_file_id is not None and not os.path.exists(self.result_file):
            output_file = self.client.files.content(batch.output_file_id)
            with open(self.result_file, 'w') as f:
                f.write(output_file.text)

    def get_file(self):
        file_response = self.client.files.content('file-AU3duZWDo2MKaauAEaiFvM')
        with open(self.result_file, "w") as f:
            f.write(file_response.text)
    
    def cancel(self):
        with open(self.folder + 'batch_id.txt', 'r') as f:
            batch_id = f.read()
        self.client.batches.cancel(batch_id)
        
    def get_cost(self):
        with open(self.result_file, 'r') as f:
            lines = f.readlines()
        
        total_cost = 0
        a = 0
        for line in lines:
            data = json.loads(line)['response']['body']
            total_cost += data['usage']['prompt_tokens'] * 1.1 / 1000000 + data['usage']['completion_tokens'] * 4.4 / 1000000
            a += data['usage']['prompt_tokens']
        print(f"Total cost: {total_cost:.6f} USD")
        print(a)
        print(len(lines))
