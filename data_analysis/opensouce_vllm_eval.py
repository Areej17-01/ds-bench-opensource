import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Optional

# Model configurations
MODEL_CONFIGS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": {
        "context_length": 8192,
        "cost_per_input_token": 0.0,  # Open source - free if self-hosted
        "cost_per_output_token": 0.0,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": {
        "context_length": 8192,
        "cost_per_input_token": 0.0,
        "cost_per_output_token": 0.0,
    }
}

class ReasoningModelEvaluator:
    def __init__(self, model_name: str, api_base_url: str, api_key: str = "EMPTY"):
        """
        Initialize evaluator for reasoning models via vLLM OpenAI-compatible API
        
        Args:
            model_name: Name of the model (must be in MODEL_CONFIGS)
            api_base_url: vLLM server URL (e.g., "http://localhost:8000/v1" or RunPod URL)
            api_key: API key (default "EMPTY" for local vLLM)
        """
        self.model_name = model_name
        self.model_config = MODEL_CONFIGS[model_name]
        
        # Initialize OpenAI client pointing to vLLM server
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url
        )
        
        self.total_cost = 0.0
        self.skipped_questions = []
        
    def read_txt(self, path: str) -> str:
        """Read text file content"""
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    
    def find_files_by_extension(self, directory: str, extensions: List[str]) -> Optional[List[str]]:
        """Find files with specific extensions in directory"""
        if not os.path.exists(directory):
            return None
        files = [f for f in os.listdir(directory) 
                if any(f.lower().endswith(ext) for ext in extensions)]
        return files if files else None
    
    def read_excel(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file"""
        xls = pd.ExcelFile(file_path)
        sheets = {}
        for sheet_name in xls.sheet_names:
            sheets[sheet_name] = xls.parse(sheet_name)
        return sheets
    
    def dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to text representation"""
        return df.to_string(index=False)
    
    def combine_sheets_text(self, sheets: Dict[str, pd.DataFrame]) -> str:
        """Combine all Excel sheets into text"""
        combined_text = ""
        for sheet_name, df in sheets.items():
            sheet_text = self.dataframe_to_text(df)
            combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
        return combined_text
    
    def get_model_response(self, text: str, temperature: float = 0.0, max_tokens: int = 4096) -> dict:
        """
        Get response from reasoning model via vLLM OpenAI-compatible API
        
        Args:
            text: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            dict with response content, tokens, cost, and time
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. I will give you a background introduction and data analysis question. You must answer the question with detailed reasoning."
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            elapsed_time = time.time() - start_time
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = (input_tokens * self.model_config["cost_per_input_token"] + 
                   output_tokens * self.model_config["cost_per_output_token"])
            
            return {
                "response": response.choices[0].message.content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": cost,
                "time": elapsed_time,
                "model": response.model
            }
            
        except Exception as e:
            print(f"Error calling model: {e}")
            return {
                "response": f"ERROR: {str(e)}",
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "time": time.time() - start_time,
                "model": self.model_name,
                "error": str(e)
            }
    
    def process_sample(self, sample: dict, data_path: str) -> List[dict]:
        """
        Process a single sample from the dataset
        
        Args:
            sample: Sample dictionary with 'id' and 'questions'
            data_path: Base path to data directory
            
        Returns:
            List of answer dictionaries
        """
        sample_id = sample["id"]
        sample_dir = os.path.join(data_path, sample_id)
        
        # Check for images - skip questions with images
        image_files = self.find_files_by_extension(sample_dir, ['.jpg', '.png', '.jpeg'])
        has_images = image_files is not None and len(image_files) > 0
        
        if has_images:
            for question_name in sample["questions"]:
                self.skipped_questions.append({
                    "folder": sample_id,
                    "question": question_name,
                    "images": image_files,
                    "reason": "Contains images - text-only evaluation"
                })
            print(f"⏭  Skipping {sample_id} - contains images: {image_files}")
            return []
        
        # Read Excel content
        excel_content = ""
        excel_files = self.find_files_by_extension(sample_dir, ['.xlsx', '.xlsb', '.xlsm'])
        
        if excel_files:
            for excel_file in excel_files:
                if "answer" in excel_file.lower():
                    continue
                excel_path = os.path.join(sample_dir, excel_file)
                try:
                    sheets = self.read_excel(excel_path)
                    combined_text = self.combine_sheets_text(sheets)
                    excel_content += f"The excel file {excel_file} is: {combined_text}\n"
                except Exception as e:
                    print(f"Warning: Could not read {excel_file}: {e}")
        
        # Read introduction
        intro_path = os.path.join(sample_dir, "introduction.txt")
        introduction = self.read_txt(intro_path) if os.path.exists(intro_path) else ""
        
        # Process each question
        answers = []
        for question_name in sample["questions"]:
            question_path = os.path.join(sample_dir, f"{question_name}.txt")
            if not os.path.exists(question_path):
                print(f"Warning: Question file not found: {question_path}")
                continue
                
            question_text = self.read_txt(question_path)
            
            # Construct prompt
            prompt = ""
            if excel_content:
                prompt += f"The workbook is detailed as follows:\n{excel_content}\n\n"
            if introduction:
                prompt += f"The introduction is detailed as follows:\n{introduction}\n\n"
            prompt += f"The question is detailed as follows:\n{question_text}"
            
            # Get model response
            result = self.get_model_response(prompt)
            
            # Add metadata
            result["sample_id"] = sample_id
            result["question_name"] = question_name
            
            answers.append(result)
            self.total_cost += result["cost"]
            
            print(f"{sample_id}/{question_name} - Tokens: {result['total_tokens']}, Time: {result['time']:.2f}s")
        
        return answers
    
    def evaluate_dataset(self, data_json_path: str, data_path: str, output_dir: str):
        """
        Evaluate entire dataset
        
        Args:
            data_json_path: Path to data.json file
            data_path: Path to data directory
            output_dir: Directory to save results
        """
        # Load samples
        samples = []
        with open(data_json_path, "r") as f:
            for line in f:
                samples.append(eval(line.strip()))
        
        print(f"Loaded {len(samples)} samples")
        print(f"Model: {self.model_name}")
        print(f" API Base: {self.client.base_url}")
        print("-" * 80)
        
        # Create output directory
        model_output_dir = os.path.join(output_dir, self.model_name.replace("/", "_"))
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Process each sample
        all_answers = []
        for sample in tqdm(samples, desc="Processing samples"):
            if len(sample.get("questions", [])) == 0:
                continue
                
            answers = self.process_sample(sample, data_path)
            
            if answers:
                # Save individual sample results
                sample_output_path = os.path.join(model_output_dir, f"{sample['id']}.json")
                with open(sample_output_path, "w") as f:
                    for answer in answers:
                        json.dump(answer, f)
                        f.write("\n")
                
                all_answers.extend(answers)
        
        # Save skipped questions
        if self.skipped_questions:
            skipped_path = os.path.join(model_output_dir, "skipped_questions.json")
            with open(skipped_path, "w") as f:
                json.dump(self.skipped_questions, f, indent=2)
            print(f"\n Skipped {len(self.skipped_questions)} questions (saved to {skipped_path})")
        
        # Save summary statistics
        summary = {
            "model": self.model_name,
            "total_samples": len(samples),
            "processed_questions": len(all_answers),
            "skipped_questions": len(self.skipped_questions),
            "total_cost": self.total_cost,
            "total_input_tokens": sum(a["input_tokens"] for a in all_answers),
            "total_output_tokens": sum(a["output_tokens"] for a in all_answers),
            "avg_time_per_question": sum(a["time"] for a in all_answers) / len(all_answers) if all_answers else 0
        }
        
        summary_path = os.path.join(model_output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n" + "=" * 80)
        print(f" Evaluation Summary")
        print(f"=" * 80)
        print(f"Processed: {summary['processed_questions']} questions")
        print(f"Skipped: {summary['skipped_questions']} questions")
        print(f"Total cost: ${summary['total_cost']:.4f}")
        print(f"Total tokens: {summary['total_input_tokens'] + summary['total_output_tokens']:,}")
        print(f"Avg time/question: {summary['avg_time_per_question']:.2f}s")
        print(f"Results saved to: {model_output_dir}")
        print("=" * 80)


# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_JSON_PATH = "./data.json"
    DATA_PATH = "./data/"
    OUTPUT_DIR = "./results"
    
    # vLLM API endpoints - UPDATE THESE WITH YOUR RUNPOD URLS
    VLLM_ENDPOINTS = {
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "http://localhost:8000/v1",  # Replace with RunPod URL
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "http://localhost:8001/v1",   # Replace with RunPod URL
    }
    
    # Choose which model to evaluate
    model_to_evaluate = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    # Create evaluator
    evaluator = ReasoningModelEvaluator(
        model_name=model_to_evaluate,
        api_base_url=VLLM_ENDPOINTS[model_to_evaluate],
        api_key="EMPTY"  # vLLM doesn't require API key by default
    )
    
    # Run evaluation
    evaluator.evaluate_dataset(
        data_json_path=DATA_JSON_PATH,
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR
    )