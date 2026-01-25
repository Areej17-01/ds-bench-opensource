import os
import re
import json
import time
import pandas as pd
from tqdm.notebook import tqdm
from openai import OpenAI

# Model configurations
MODEL_LIMITS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 128_000,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 128_000,
}

# The cost per token for each model input (free for self-hosted)
MODEL_COST_PER_INPUT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 0.0,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.0,
}

# The cost per token for each model output (free for self-hosted)
MODEL_COST_PER_OUTPUT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 0.0,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.0,
}


def get_model_response(client, text, model):
    """Get response from vLLM server using OpenAI-compatible API"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a data analyst. I will give you a background introduction and data analysis question. You must answer the question."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response


def find_jpg_files(directory):
    if not os.path.exists(directory):
        return None
    jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else None


def find_excel_files(directory):
    if not os.path.exists(directory):
        return None
    excel_files = [file for file in os.listdir(directory) if (file.lower().endswith('xlsx') or file.lower().endswith('xlsb') or file.lower().endswith('xlsm')) and not "answer" in file.lower()]
    return excel_files if excel_files else None


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


# Initialize vLLM client
# Update this URL with your RunPod endpoint
VLLM_API_BASE = "http://localhost:8000/v1"  # Replace with your RunPod URL
client = OpenAI(api_key="EMPTY", base_url=VLLM_API_BASE)

# Select model to evaluate
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

data_path = "./data/"
total_cost = 0
skipped_questions = []

# Load samples
samples = []
with open("./data.json", "r") as f:
    for line in f:
        samples.append(eval(line.strip()))

for id in tqdm(range(len(samples))):
    sample = samples[id]
    if len(sample["questions"]) > 0:
        
        # Check for images - skip if images present
        image = find_jpg_files(os.path.join(data_path, sample["id"]))
        if image:
            for question_name in sample["questions"]:
                skipped_questions.append({
                    "folder": sample["id"],
                    "question": question_name,
                    "images": image,
                    "reason": "Contains images"
                })
            print(f"Skipping {sample['id']} - contains images: {image}")
            continue
        
        excel_content = ""
        excels = find_excel_files(os.path.join(data_path, sample["id"]))
        if excels:
            for excel in excels:
                excel_file_path = os.path.join(data_path, sample["id"], excel)
                try:
                    sheets = read_excel(excel_file_path)
                    combined_text = combine_sheets_text(sheets)
                    excel_content += f"The excel file {excel} is: " + combined_text
                except Exception as e:
                    print(f"Error reading {excel_file_path}: {e}")

        introduction = read_txt(os.path.join(data_path, sample["id"], "introduction.txt"))
        questions = []
        for question_name in sample["questions"]:
            questions.append(read_txt(os.path.join(data_path, sample["id"], question_name+".txt")))
        
        text = ""
        if excel_content:
            text += f"The workbook is detailed as follows. {excel_content} \n"
        text += f"The introduction is detailed as follows. \n {introduction} \n"
        answers = []
        for question in questions:
            prompt = text + f"The questions are detailed as follows. \n {question}"
            
            start = time.time()
            response = get_model_response(client, prompt, model)
            cost = response.usage.completion_tokens * MODEL_COST_PER_OUTPUT[model] + response.usage.prompt_tokens * MODEL_COST_PER_INPUT[model]
            
            answers.append({
                "id": sample["id"], 
                "model": response.model, 
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens, 
                "cost": cost, 
                "time": time.time()-start, 
                "response": response.choices[0].message.content
            })
            total_cost += cost
            print("Total cost: ", total_cost)
            
        save_path = os.path.join("./save_process", model.replace("/", "_"))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, sample['id']+".json"), "w") as f:
            for answer in answers:
                json.dump(answer, f)
                f.write("\n")

# Save skipped questions
if skipped_questions:
    skip_save_path = os.path.join("./save_process", model.replace("/", "_"))
    if not os.path.exists(skip_save_path):
        os.makedirs(skip_save_path)
    with open(os.path.join(skip_save_path, "skipped_questions.json"), "w") as f:
        json.dump(skipped_questions, f, indent=2)
    print(f"Skipped {len(skipped_questions)} questions with images")