import os
import re
import json
import time
import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Model configurations
MODEL_LIMITS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 128_000,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 128_000,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 128_000,
}

# The cost per token for each model input (free for self-hosted)
MODEL_COST_PER_INPUT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 0.0,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.0,
}

# The cost per token for each model output (free for self-hosted)
MODEL_COST_PER_OUTPUT = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": 0.0,
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.0,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.0,
}


def count_tokens(text, tokenizer):
    """Count tokens using the actual model tokenizer"""
    return len(tokenizer.encode(text))


def truncate_text(text, max_tokens, tokenizer):
    """Truncate text to fit within token limit using actual tokenizer"""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        # Keep the last max_tokens to preserve most recent context
        text = tokenizer.decode(tokens[-max_tokens:], skip_special_tokens=True)
    return text


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


# Select model to evaluate
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Initialize the actual tokenizer for the model
print(f"Loading tokenizer for {model}...")
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
print("Tokenizer loaded successfully!")

# Initialize vLLM with offline inference
print(f"Loading vLLM model for offline inference...")
print("This will take several minutes...")

# For single GPU:
llm = LLM(
    model=model,
    max_model_len=131072,
    gpu_memory_utilization=0.95,
    trust_remote_code=True
)

# # For 4 GPUs with tensor parallelism :
# llm = LLM(
#     model=model,
#     tensor_parallel_size=4,  # Use 4 GPUs together
#     max_model_len=131072,
#     gpu_memory_utilization=0.95,
#     trust_remote_code=True
# )

print("Model loaded successfully!")

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=16384,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)

# Reserve tokens for generation
tokens4generation = 16384

# Calculate actual max input tokens (with safety margin)
SAFETY_MARGIN = 1000
max_input_tokens = MODEL_LIMITS[model] - tokens4generation - SAFETY_MARGIN

data_path = "./data/"
total_cost = 0
skipped_questions = []

# Load samples
samples = []
with open("./data.json", "r") as f:
    for line in f:
        samples.append(eval(line.strip()))

print(f"Processing {len(samples)} samples...")

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
            
            # Count tokens with actual tokenizer
            current_tokens = count_tokens(prompt, tokenizer)
            print(f"Current prompt tokens: {current_tokens}, Max allowed: {max_input_tokens}")
            
            # Truncate text if it exceeds model's context limit
            if current_tokens > max_input_tokens:
                print(f"Truncating from {current_tokens} to {max_input_tokens} tokens")
                cut_text = truncate_text(prompt, max_input_tokens, tokenizer)
                # Verify truncation
                verify_tokens = count_tokens(cut_text, tokenizer)
                print(f"After truncation: {verify_tokens} tokens")
            else:
                cut_text = prompt
            
            # Format prompt for offline inference with system message
            formatted_prompt = f"""You are a data analyst. I will give you a background introduction and data analysis question. You must answer the question.

{cut_text}"""
            
            start = time.time()
            
            # Generate response using offline vLLM
            outputs = llm.generate([formatted_prompt], sampling_params)
            
            elapsed_time = time.time() - start
            
            # Extract response
            response_text = outputs[0].outputs[0].text
            
            # Get token counts from output
            prompt_tokens = len(outputs[0].prompt_token_ids)
            completion_tokens = len(outputs[0].outputs[0].token_ids)
            
            cost = completion_tokens * MODEL_COST_PER_OUTPUT[model] + prompt_tokens * MODEL_COST_PER_INPUT[model]
            
            answers.append({
                "id": sample["id"], 
                "model": model, 
                "input": prompt_tokens,
                "output": completion_tokens, 
                "cost": cost, 
                "time": elapsed_time, 
                "response": response_text
            })
            total_cost += cost
            print(f"Question completed in {elapsed_time/60:.2f} minutes")
            print(f"Total cost: {total_cost}")
            
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

print(f"\nAll processing complete!")
print(f"Total time: {time.time()/3600:.2f} hours")