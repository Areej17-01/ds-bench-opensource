import os
import json
import base64
import tiktoken
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# Model configuration
MODEL_LIMITS = {
    "gpt-3.5-turbo-0125": 16_385,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4o-2024-05-13": 128_000
}

MODEL_COST_PER_INPUT = {
    "gpt-3.5-turbo-0125": 0.0000005,
    "gpt-4-turbo-2024-04-09": 0.00001,
    "gpt-4o-2024-05-13": 0.000005
}

MODEL_COST_PER_OUTPUT = {
    "gpt-3.5-turbo-0125": 0.0000015,
    "gpt-4-turbo-2024-04-09": 0.00003,
    "gpt-4o-2024-05-13": 0.000015
}

# Helper functions
def find_jpg_files(directory):
    """Find JPG/PNG files in directory."""
    if not os.path.exists(directory):
        return None
    jpg_files = [file for file in os.listdir(directory) 
                 if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else None

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_excel_files(directory):
    """Find Excel files in directory."""
    if not os.path.exists(directory):
        return None
    excel_files = [file for file in os.listdir(directory) 
                   if (file.lower().endswith('xlsx') or 
                       file.lower().endswith('xlsb') or 
                       file.lower().endswith('xlsm')) 
                   and "answer" not in file.lower()]
    return excel_files if excel_files else None

def read_excel(file_path):
    """Read all sheets from Excel file."""
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets

def dataframe_to_text(df):
    """Convert DataFrame to text."""
    return df.to_string(index=False)

def combine_sheets_text(sheets):
    """Combine all sheet texts."""
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def read_txt(path):
    """Read text file."""
    with open(path, "r", encoding='utf-8') as f:
        return f.read()

def get_gpt_res(text, image, model, client):
    """Get GPT response - text only (image is None for text-only mode)."""
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
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response

# Main execution
if __name__ == "__main__":
    # Configuration
    client = OpenAI(api_key="your-api-key")  # Set your API key here
    tokens4generation = 6000
    model = "gpt-4-turbo-2024-04-09"  # or "gpt-3.5-turbo-0125" or "gpt-4o-2024-05-13"
    data_path = "./data/"
    total_cost = 0
    encoding = tiktoken.encoding_for_model(model)
    
    # Load samples
    samples = []
    with open("./data.json", "r") as f:
        for line in f:
            samples.append(eval(line.strip()))
    
    # Initialize skipped questions file
    skipped_questions = []
    skipped_file = "skipped_questions.json"
    
    # Process each sample
    for sample in tqdm(samples, desc="Processing samples"):
        if len(sample["questions"]) == 0:
            continue
        
        sample_dir = os.path.join(data_path, sample["id"])
        
        # Check for images - if found, skip all questions for this sample
        image_files = find_jpg_files(sample_dir)
        if image_files:
            # Skip this sample and log to skipped_questions.json
            skipped_entry = {
                "id": sample["id"],
                "folder_name": sample["id"],
                "images": image_files,
                "image_paths": [os.path.join(sample_dir, img) for img in image_files],
                "questions": sample["questions"],
                "reason": "Image found in folder"
            }
            skipped_questions.append(skipped_entry)
            
            # Save skipped questions immediately
            with open(skipped_file, "w", encoding='utf-8') as f:
                json.dump(skipped_questions, f, indent=2, ensure_ascii=False)
            
            print(f"Skipped sample {sample['id']} - found images: {image_files}")
            continue
        
        # No image found - process normally
        excel_content = ""
        excels = find_excel_files(sample_dir)
        if excels:
            for excel in excels:
                excel_file_path = os.path.join(sample_dir, excel)
                try:
                    sheets = read_excel(excel_file_path)
                    combined_text = combine_sheets_text(sheets)
                    excel_content += f"The excel file {excel} is: " + combined_text
                except Exception as e:
                    print(f"Error reading Excel file {excel_file_path}: {e}")
        
        # Read introduction
        introduction_path = os.path.join(sample_dir, "introduction.txt")
        if not os.path.exists(introduction_path):
            print(f"Warning: introduction.txt not found for {sample['id']}")
            continue
        
        introduction = read_txt(introduction_path)
        
        # Read questions
        questions = []
        question_names = []
        for question_name in sample["questions"]:
            question_path = os.path.join(sample_dir, question_name + ".txt")
            if os.path.exists(question_path):
                questions.append(read_txt(question_path))
                question_names.append(question_name)
            else:
                print(f"Warning: {question_name}.txt not found for {sample['id']}")
        
        # Prepare text content
        text = ""
        if excel_content:
            text += f"The workbook is detailed as follows. {excel_content} \n"
        text += f"The introduction is detailed as follows. \n {introduction} \n"
        
        # Process each question
        answers = []
        for question, question_name in zip(questions, question_names):
            prompt = text + f"The questions are detailed as follows. \n {question}"
            
            # Truncate text if needed
            try:
                encoded = encoding.encode(prompt)
                if len(encoded) > MODEL_LIMITS[model] - tokens4generation:
                    cut_text = encoding.decode(encoded[-(MODEL_LIMITS[model] - tokens4generation):])
                else:
                    cut_text = prompt
            except Exception as e:
                print(f"Error encoding prompt for {sample['id']}: {e}")
                cut_text = prompt
            
            # Call GPT API
            try:
                start_time = time.time()
                response = get_gpt_res(cut_text, None, model, client)  # image=None for text-only
                elapsed_time = time.time() - start_time
                
                cost = (response.usage.completion_tokens * MODEL_COST_PER_OUTPUT[model] + 
                       response.usage.prompt_tokens * MODEL_COST_PER_INPUT[model])
                
                answer = {
                    "id": sample["id"],
                    "question": question_name,
                    "model": response.model,
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                    "cost": cost,
                    "time": elapsed_time,
                    "response": response.choices[0].message.content
                }
                answers.append(answer)
                total_cost += cost
                
                print(f"Processed {sample['id']}/{question_name} - Cost: ${cost:.6f}, Total: ${total_cost:.6f}")
                
            except Exception as e:
                print(f"Error processing {sample['id']}/{question_name}: {e}")
                time.sleep(5)  # Wait before retrying
                continue
        
        # Save answers
        if answers:
            save_path = os.path.join("./save_process", model)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            result_file = os.path.join(save_path, sample['id'] + ".json")
            with open(result_file, "w", encoding='utf-8') as f:
                for answer in answers:
                    json.dump(answer, f, ensure_ascii=False)
                    f.write("\n")
    
    print(f"\n=== Summary ===")
    print(f"Total cost: ${total_cost:.6f}")
    print(f"Skipped {len(skipped_questions)} samples with images")
    print(f"Skipped questions saved to: {skipped_file}")
