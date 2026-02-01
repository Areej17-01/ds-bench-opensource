

import os
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, UserProxyAgent
from IPython.display import Image, display
import json
import base64
import re
import time
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Utility functions
def gpt_tokenize(string: str, encoding) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens

def find_jpg_files(directory):
    jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_excel_files(directory):
    jpg_files = [file for file in os.listdir(directory) if (file.lower().endswith('xlsx') or file.lower().endswith('xlsb') or file.lower().endswith('xlsm')) and not "answer" in file.lower()]
    return jpg_files if jpg_files else None

def read_excel(file_path):
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets

def dataframe_to_text(df):
    text = df.to_string(index=False)
    return text

def combine_sheets_text(sheets):
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def read_txt(path):
    with open(path, "r") as f:
        return f.read()

def truncate_text(text, max_tokens=128000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        text = ' '.join(tokens[-max_tokens:])
    return text

MODEL_LIMITS = {
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "meta-llama/Llama-3.1-8B-Instruct":128_000,
}

MODEL_COST_PER_INPUT = {
    "gpt-3.5-turbo-0125": 0.0000005,
    "gpt-4-turbo-2024-04-09": 0.00001,
    "gpt-4o-2024-05-13": 0.000005,
    "gpt-4o-mini-2024-07-18": 0.00000015,
    "meta-llama/Llama-3.1-8B-Instruct":0.0,
}

MODEL_COST_PER_OUTPUT = {
    "gpt-3.5-turbo-0125": 0.0000015,
    "gpt-4-turbo-2024-04-09": 0.00003,
    "gpt-4o-2024-05-13": 0.000015,
    "gpt-4o-mini-2024-07-18": 0.0000006,
    "meta-llama/Llama-3.1-8B-Instruct":0.0,
}

MODEL="meta-llama/Llama-3.1-8B-Instruct"

print("Loading tokenizer and vLLM model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)

vllm_model = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=MODEL_LIMITS[MODEL],
)

vllm_sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8000,
    top_p=1.0,
)
print("Model loaded successfully!")

# Proper classes for pickling
class Message:
    def __init__(self, content, role='assistant'):
        self.content = content
        self.role = role

class Choice:
    def __init__(self, message):
        self.message = message

class Usage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

class ModelResponse:
    def __init__(self, content):
        self.choices = [Choice(Message(content))]
        self.usage = Usage()
        self.cost = 0

class VLLMClient:
    def __init__(self, config, **kwargs):
        self.model_name = config["model"]
        print(f"VLLMClient initialized for model: {self.model_name}")

    def create(self, params):
        if params.get("stream", False) and "messages" in params:
            raise NotImplementedError("Streaming is not supported for vLLM")

        messages = params["messages"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = vllm_model.generate([prompt], vllm_sampling_params)
        output_text = outputs[0].outputs[0].text
        
        # Calculate actual token counts using the tokenizer
        prompt_tokens = len(tokenizer.encode(prompt))
        completion_tokens = len(tokenizer.encode(output_text))
        
        # Create response and populate token counts
        response = ModelResponse(output_text)
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        response.usage.total_tokens = prompt_tokens + completion_tokens
        
        return response

    def message_retrieval(self, response):
        return [response.choices[0].message.content]

    def cost(self, response) -> float:
        return 0

    @staticmethod
    def get_usage(response):
        # CRITICAL: AutoGen expects a dictionary with ALL required keys
        # Including "cost" and "model" in addition to token counts
        return {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'total_tokens': response.usage.total_tokens,
            'cost': 0,  # Free for self-hosted
            'model': 'meta-llama/Llama-3.1-8B-Instruct'  # Must match the model name
        }

samples = []
with open("./data.json", "r") as f:
    for line in f:
        samples.append(eval(line.strip()))

print(f"Loaded {len(samples)} samples")

def get_response(text, config_list):
    assistant = autogen.AssistantAgent(
        name="assistant",
        llm_config={
            "cache_seed": 41,
            "config_list": config_list,
            "temperature": 0,
        },
    )
    assistant.register_model_client(model_client_cls=VLLMClient)
    
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
        code_execution_config={
            "executor": LocalCommandLineCodeExecutor(work_dir="coding"),
        },
    )
    
    chat_res = user_proxy.initiate_chat(assistant, message=text, summary_method="reflection_with_llm")
    return chat_res

model = MODEL
config_list = [{"model": model, "model_client_cls": "VLLMClient"}]
total_cost = 0
skipped_questions = []
data_path="./data/"
for id in tqdm(range(0, (len(samples)))):
    print(f"running folder {id}")
    sample = samples[id]
    if len(sample["questions"]) > 0:
        image = find_jpg_files(os.path.join(data_path, sample["id"]))
        
        if image:
            print(f"Skipping sample {sample['id']} - contains images")
            skipped_questions.append({"id": sample["id"], "reason": "contains_image", "images": image})
            continue
        
        excels = find_excel_files(os.path.join(data_path, sample["id"]))
        introduction = read_txt(os.path.join(data_path, sample["id"], "introduction.txt"))
        questions = []
        for question_name in sample["questions"]:
            questions.append(read_txt(os.path.join(data_path, sample["id"], question_name+".txt")))
        
        text = f"The introduction is detailed as follows. \n {introduction}" 
        if excels:
            text += "\n \n The worksheet can be obtained in the path: "
            for excel in excels:
                text += f" {os.path.abspath(os.path.join(data_path, sample['id'], excel))}"
        
        answers = []
        for question in tqdm(questions, desc="Processing questions"):
            all_context = text + f"The question is detailed as follows. \n {question} \nPlease answer the question. "
            start = time.time()
            
            try:
                response = get_response(all_context, config_list)
                summary = response.summary
                history = response.chat_history
                
                # Extract token counts from AutoGen's cost tracking (same way as GPT)
                input_tokens = 0
                output_tokens = 0
                if hasattr(response, 'cost') and response.cost:
                    # For custom clients, AutoGen stores usage in response.cost
                    if 'usage_including_cached_inference' in response.cost:
                        # Get usage for this model
                        if model in response.cost['usage_including_cached_inference']:
                            input_tokens = response.cost['usage_including_cached_inference'][model].get('prompt_tokens', 0)
                            output_tokens = response.cost['usage_including_cached_inference'][model].get('completion_tokens', 0)
                            print(f"SUCCESS: Found tokens - input: {input_tokens}, output: {output_tokens}")
                else:
                    print("WARNING: response.cost not available")
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(3)
                history = "I cannot solve this task."
                summary = "I cannot solve this task."
                input_tokens = 0
                output_tokens = 0
            
            print(f"Total cost: {total_cost}")
            print(f"Token usage - Input: {input_tokens}, Output: {output_tokens}")
            
            answers.append({
                "id": sample["id"], 
                "model": model, 
                "input": input_tokens, 
                "output": output_tokens, 
                "cost": 0, 
                "time": time.time()-start, 
                'summary': summary, 
                "history": history
            })
    
    save_path = os.path.join("./save_process", f"{MODEL}-autoagent")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, sample['id'] + ".json"), "w") as f:
        for answer in answers:
            json.dump(answer, f)
            f.write("\n")

with open("./skipped_questions.json", "w") as f:
    json.dump(skipped_questions, f, indent=2)

print(f"\nTotal samples processed: {len(samples) - len(skipped_questions)}")
print(f"Total samples skipped: {len(skipped_questions)}")
print("Done!")