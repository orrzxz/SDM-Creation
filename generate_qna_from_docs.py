import os
import glob
import json
from typing import List, Dict, Optional, Any, Tuple
from transformers import AutoTokenizer
from sentence_transformers import CrossEncoder
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

# --- Configuration ---
HOUDINI_DOCS_MKDOWN_DIR = "houdini_docs_mkdown"
ODFORCE_SCRAPMD_DIR = "odforce_scrapMD"
OUTPUT_ROOT = "qna_dataset"
RAW_MD_DIR = os.path.join(OUTPUT_ROOT, "raw_md_files")
PROCESSED_DIR = os.path.join(OUTPUT_ROOT, "processed_text")
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "output")

# Update model names if you want to use different models
SUITABILITY_MODEL = "cross-encoder/nli-deberta-v3-base"
QNA_MODEL = "google/flan-t5-xxl"
MIN_SUITABILITY_SCORE = 0.7
MAX_INPUT_LENGTH = 1000  # Truncate longer documents to manage memory for prompt creation

# Batching and Model Parameters
SUITABILITY_BATCH_SIZE = 32
QNA_GENERATION_MAX_NEW_TOKENS = 512
VLLM_MAX_MODEL_LEN = 2048 # Max sequence length for vLLM (prompt + generated tokens)

# Initialize models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Suitability classifier (zero-shot)
suitability_model = CrossEncoder(SUITABILITY_MODEL, max_length=512, device=device)

# Initialize QnA model with vLLM
if device == "cuda":
    print(f"Initializing vLLM for QnA model: {QNA_MODEL}")
    qna_model_vllm = LLM(
        model=QNA_MODEL,
        tokenizer=QNA_MODEL,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        max_model_len=VLLM_MAX_MODEL_LEN,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=QNA_GENERATION_MAX_NEW_TOKENS
    )
else:
    print("Warning: vLLM typically requires a CUDA GPU. QnA generation will be skipped or may fail.")
    qna_model_vllm = None
    sampling_params = None

# --- Helper Functions ---

def find_markdown_files(directories: List[str]) -> List[str]:
    """Finds all markdown (.md) files in the given directories."""
    md_files = []
    for directory in directories:
        if not os.path.isdir(directory):
            print(f"Warning: Directory not found - {directory}")
            continue
        # Recursively find all .md files
        for filepath in glob.glob(os.path.join(directory, "**", "*.md"), recursive=True):
            md_files.append(filepath)
    return md_files

def read_file_content(file_path: str) -> Optional[str]:
    """Reads the content of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None
    except UnicodeDecodeError as e:
        print(f"Unicode decode error in file {file_path}: {e}. Skipping file.")
        return None

# --- Updated LLM Functions ---
def is_file_suitable_for_qna(file_content: str, file_path: str) -> bool:
    """Determine processing strategy based on source folder"""
    if file_path.startswith(HOUDINI_DOCS_MKDOWN_DIR):
        # Always process Houdini documentation files
        return True
    else:
        # Use original suitability check for forum files
        if len(file_content) < 100:
            return False
        pairs = [[file_content[:MAX_INPUT_LENGTH], "This text contains questions and answers"]]
        scores = suitability_model.predict(pairs)
        return scores[0] >= MIN_SUITABILITY_SCORE

def parse_qna_response(response: str) -> List[Dict[str, str]]:
    """Parse model output into structured QnA pairs"""
    qna_pairs = []
    current_q = None
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith("Q:"):
            current_q = line[2:].strip()
        elif line.startswith("A:") and current_q:
            qna_pairs.append({
                "question": current_q,
                "answer": line[2:].strip()
            })
            current_q = None
            
    return qna_pairs

# --- Modified Helper Functions ---
def get_output_path(source_file: str) -> str:
    """Preserve source directory structure in output"""
    for source_dir in [HOUDINI_DOCS_MKDOWN_DIR, ODFORCE_SCRAPMD_DIR]:
        if source_file.startswith(source_dir):
            relative_path = os.path.relpath(source_file, source_dir)
            return os.path.join(OUTPUT_DIR, source_dir, relative_path)
    return os.path.join(OUTPUT_DIR, "unknown_source", os.path.basename(source_file))

def ensure_directory_exists(file_path: str):
    """Create directories needed for a file path"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

# --- New Batch Processing Helper Functions ---

def batch_check_suitability(
    odforce_files_data: List[Tuple[str, str]], # List of (path, content)
    model: CrossEncoder,
    min_score: float,
    max_len_truncate: int, # Max length for content part of the pair
    batch_size: int
) -> List[Tuple[str, str]]:
    """
    Checks suitability of Odforce forum posts in batches.
    Returns a list of (path, content) for suitable files.
    """
    if not odforce_files_data:
        return []

    suitable_files_data = []
    
    # Prepare all pairs for the model
    # Each item for the CrossEncoder is a pair: [text1, text2]
    suitability_input_pairs = []
    original_indices = [] # To map scores back to original file data

    for idx, (path, content) in enumerate(odforce_files_data):
        # content is already pre-filtered for min length (e.g. 100 chars)
        suitability_input_pairs.append([content[:max_len_truncate], "This text contains questions and answers"])
        original_indices.append(idx)
        
    if not suitability_input_pairs:
        return []

    print(f"Running suitability check for {len(suitability_input_pairs)} Odforce files in batches of {batch_size}...")
    
    all_scores = model.predict(
        suitability_input_pairs, 
        batch_size=batch_size, 
        show_progress_bar=True
    )

    for i, score in enumerate(all_scores):
        if score >= min_score:
            original_file_idx = original_indices[i]
            suitable_files_data.append(odforce_files_data[original_file_idx])
            
    print(f"Found {len(suitable_files_data)} suitable Odforce files out of {len(odforce_files_data)}.")
    return suitable_files_data

# --- New vLLM Batch QnA Generation Function ---
def batch_generate_qna_vllm(
    files_for_qna_data: List[Tuple[str, str]], # List of (path, content)
    model: LLM, # vLLM's LLM object
    sampling_params_vllm: SamplingParams, # vLLM's sampling parameters
    prompt_content_max_len: int, # MAX_INPUT_LENGTH for content truncation in prompt
) -> List[Tuple[str, List[Dict[str, str]]]]: # Returns list of (path, qna_list)
    """
    Generates Q&A pairs in batches for a list of files (path, content) using vLLM.
    """
    if not files_for_qna_data or model is None:
        if model is None:
            print("vLLM model not initialized (likely no CUDA GPU). Skipping QnA generation.")
        return []

    all_generated_qna_results = [] # To store (path, parsed_qna_list)
    
    prompts_with_metadata = []
    for file_path, file_content in files_for_qna_data:
        if file_path.startswith(HOUDINI_DOCS_MKDOWN_DIR):
            prompt = f"""Create comprehensive technical Q&A pairs from this documentation. Rules:
1. Generate relevant questions based on the documentation content
2. Answers must be factual and directly from the text
2. Use technical terminology precisely
3. Format as: Q: [question] A: [answer]

Documentation: {file_content[:prompt_content_max_len]}"""
        else: # Odforce file
            prompt = f"""Read this forum discussion and generate question-answer pairs. Rules:
1. Only generate questions actually answered in the text
2. Answers must summarize the solution clearly
3. Format as: Q: [question] A: [answer]
4. Skip speculative or unanswered questions

Discussion: {file_content[:prompt_content_max_len]}"""
        prompts_with_metadata.append({"prompt": prompt, "path": file_path})

    if not prompts_with_metadata:
        return []

    print(f"Generating Q&A for {len(prompts_with_metadata)} files using vLLM...")
    
    list_of_prompts = [item["prompt"] for item in prompts_with_metadata]
    original_paths = [item["path"] for item in prompts_with_metadata]
    
    try:
        request_outputs = model.generate(list_of_prompts, sampling_params_vllm, use_tqdm=True)
        
        for i, req_output in enumerate(request_outputs):
            original_file_path = original_paths[i]
            if req_output.outputs:
                response_text = req_output.outputs[0].text 
                qna_pairs_for_file = parse_qna_response(response_text)
                all_generated_qna_results.append((original_file_path, qna_pairs_for_file))
                
                if not qna_pairs_for_file:
                    pass
            else:
                print(f"Warning: No output generated by vLLM for {original_file_path}")
                all_generated_qna_results.append((original_file_path, []))
                
    except Exception as e:
        print(f"Error during vLLM Q&A generation: {e}")
        for path_in_failed_batch in original_paths:
             all_generated_qna_results.append((path_in_failed_batch, []))

    return all_generated_qna_results

# --- Updated Main Function ---
def main():
    print(f"Starting QnA dataset generation process using device: {device} for suitability model.")
    if qna_model_vllm is None and device == "cuda":
        print("vLLM QnA model failed to initialize despite CUDA being available. Check vLLM setup.")
    elif qna_model_vllm is None:
        print("QnA generation will be skipped as vLLM model is not available (e.g. no CUDA).")

    input_dirs = [HOUDINI_DOCS_MKDOWN_DIR, ODFORCE_SCRAPMD_DIR]
    
    for dir_path in [OUTPUT_ROOT, RAW_MD_DIR, PROCESSED_DIR, OUTPUT_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    markdown_files_paths = find_markdown_files(input_dirs)
    if not markdown_files_paths:
        print("No markdown files found. Exiting.")
        return
    print(f"Found {len(markdown_files_paths)} markdown files to process.")

    # Load content and categorize
    houdini_files_data: List[Tuple[str, str]] = []
    odforce_files_data: List[Tuple[str, str]] = []

    for md_file_path in tqdm(markdown_files_paths, desc="Reading files"):
        content = read_file_content(md_file_path)
        if not content:
            print(f"Warning: Could not read content from {md_file_path}")
            continue
        
        if md_file_path.startswith(HOUDINI_DOCS_MKDOWN_DIR):
            houdini_files_data.append((md_file_path, content))
        elif md_file_path.startswith(ODFORCE_SCRAPMD_DIR):
            if len(content) >= 100: # Basic pre-filter for suitability length
                odforce_files_data.append((md_file_path, content))
    
    print(f"Read {len(houdini_files_data)} Houdini docs and {len(odforce_files_data)} potential Odforce files.")

    # Batch suitability check for Odforce files
    suitable_odforce_files_data = batch_check_suitability(
        odforce_files_data,
        suitability_model,
        MIN_SUITABILITY_SCORE,
        MAX_INPUT_LENGTH, # Truncate content for suitability check input
        SUITABILITY_BATCH_SIZE
    )

    # Combine Houdini docs and suitable Odforce files for QnA generation
    all_files_for_qna_data = houdini_files_data + suitable_odforce_files_data
    
    if not all_files_for_qna_data:
        print("No files deemed suitable for QnA generation after filtering. Exiting.")
        return
    
    print(f"Total files for QnA generation: {len(all_files_for_qna_data)}")

    # Batch generate Q&A pairs
    results_from_qna_generation = []
    if qna_model_vllm and sampling_params and all_files_for_qna_data:
        results_from_qna_generation = batch_generate_qna_vllm(
            all_files_for_qna_data,
            qna_model_vllm,
            sampling_params,
            MAX_INPUT_LENGTH, # For content truncation in prompt
        )
    elif not all_files_for_qna_data:
        print("No suitable files to process for QnA generation.")
    else:
        print("Skipping QnA generation as vLLM model is not available or no files to process.")
        

    master_dataset = []
    processed_files_count = 0
    generated_qna_count = 0

    for original_file_path, qna_pairs_list in tqdm(results_from_qna_generation, desc="Saving results"):
        processed_files_count +=1 # Counts files attempted for QnA generation
        
        output_path_md_equivalent = get_output_path(original_file_path)
        # Ensure the base directory for the JSON file exists
        ensure_directory_exists(output_path_md_equivalent) # Creates dir for the .md equivalent path
        output_json_path = os.path.splitext(output_path_md_equivalent)[0] + ".json"
        
        if qna_pairs_list:
            generated_qna_count += len(qna_pairs_list)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source": original_file_path,
                    "qna_pairs": qna_pairs_list
                }, f, indent=2)
            
            master_dataset.extend([{
                **pair,
                "source_file": original_file_path,
                "source_dir": os.path.dirname(original_file_path)
            } for pair in qna_pairs_list])
        else:
            print(f"No QnA pairs were generated or parsed for {os.path.basename(original_file_path)}.")


    # Save master dataset
    master_file = os.path.join(OUTPUT_ROOT, "full_dataset.json")
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(master_dataset, f, indent=2)
    
    print(f"--- Summary ---")
    print(f"Initial markdown files found: {len(markdown_files_paths)}")
    print(f"Files processed for QnA generation: {processed_files_count} (out of {len(all_files_for_qna_data)} candidates after suitability)")
    print(f"Total QnA pairs generated and saved: {len(master_dataset)} (from {generated_qna_count} raw generated pairs before any losses)")
    print(f"Master dataset saved to: {master_file}")

if __name__ == "__main__":
    main() 