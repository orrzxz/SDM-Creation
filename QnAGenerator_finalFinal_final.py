#!/usr/bin/env python3

import os
import json
import csv
import re
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Tuple
from openai import OpenAI
from tqdm import tqdm
import torch
from dataclasses import dataclass
import threading

# Constants for file paths and configurations
HOUDINI_DOCS_MKDOWN = 'houdini_docs_mkdown'
ODFORCE_SCRAPMD = 'odforce_scrapMD'
CHUNKED_MARKDOWN_DIR = 'Chunked_Markdown'
QNA_DATASET_DIR = 'qna_dataset'

# Multi-Model LM Studio Configuration
MODEL_CONFIGS = [
    {
        "name": "qwen3-4b-mlx",
        "base_url": "http://192.168.1.72:1234/v1",
        "api_key": "lm-studio",
        "port": 1234
    },
    {
        "name": "qwen3-4b-mlx:2", 
        "base_url": "http://192.168.1.72:1234/v1",  # Same URL, different model identifier
        "api_key": "lm-studio",
        "port": 1234
    },
    {
        "name": "qwen3-4b", 
        "base_url": "http://localhost:1234/v1",  # Same URL, different model identifier
        "api_key": "lm-studio",
        "port": 1234
    },
    # {
    #     "name": "qwen3-4b:2", 
    #     "base_url": "http://localhost:1234/v1",  # Same URL, different model identifier
    #     "api_key": "lm-studio",
    #     "port": 1234
    # }
    # Add more model configurations as needed
]

BATCH_SIZE = 32  # Reduced for API calls
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 1024  # Increased from 512 to allow longer, more detailed answers
MAX_WORKERS = len(MODEL_CONFIGS)  # Number of concurrent workers
USE_CONCURRENT_PROCESSING = True  # Set to False for sequential processing

# Performance monitoring
@dataclass
class BatchMetrics:
    total_prompts: int
    total_time: float
    tokens_per_second: float
    average_latency: float
    batch_size_used: int
    model_name: str

# Global variables for OpenAI clients
clients = {}
client_lock = threading.Lock()

def setup_lm_studio_clients():
    """Set up OpenAI clients to connect to multiple LM Studio instances"""
    global clients
    
    print(f"Initializing LM Studio clients for {len(MODEL_CONFIGS)} models...")
    
    successful_clients = {}
    created_clients = {}  # Track clients by base_url to avoid duplicates
    
    for i, config in enumerate(MODEL_CONFIGS):
        model_name = config["name"]
        base_url = config["base_url"]
        api_key = config["api_key"]
        
        print(f"Setting up model {i+1}/{len(MODEL_CONFIGS)}: {model_name}")
        print(f"  Base URL: {base_url}")
        
        try:
            # Check if we already have a client for this base_url
            if base_url in created_clients:
                client = created_clients[base_url]
                print(f"  Using existing client for {base_url}")
            else:
                # Initialize OpenAI client pointing to LM Studio
                client = OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
                created_clients[base_url] = client
                print(f"  Created new client for {base_url}")
            
            # Test connection by listing models
            models = client.models.list()
            available_models = [model.id for model in models.data]
            print(f"  Available models: {available_models}")
            
            if model_name not in available_models:
                print(f"  Warning: Model '{model_name}' not found. Available: {available_models}")
                if available_models:
                    # Ask user to choose or use first available
                    print(f"  Skipping model '{model_name}' - not available")
                    continue
                else:
                    print(f"  No models available, skipping this model")
                    continue
            
            successful_clients[model_name] = {
                'client': client,
                'config': config
            }
            print(f"  ✓ Model {model_name} configured successfully!")
            
        except Exception as e:
            print(f"  ✗ Error configuring model {model_name}: {e}")
            print(f"  Make sure LM Studio instance is running on port {config['port']}")
            continue
    
    clients = successful_clients
    
    if not clients:
        print("No LM Studio models could be configured!")
        print("Please ensure:")
        print("1. LM Studio instance is running on the specified port")
        print("2. API server is enabled in LM Studio")
        print("3. Models are loaded in the instance")
        print("4. Model names match exactly (including any :2, :3 suffixes)")
        return False
    
    print(f"Successfully configured {len(clients)} models!")
    return True

# Prompt Templates
PROMPT_TEMPLATES = {
    'houdini_docs': """You are tasked with generating **question-and-answer pairs** based on the content of a provided Houdini documentation file.

#### General Instructions:
- Generate **3-5 natural language questions** and their corresponding **answers**.
- Use **human-style phrasing** for questions — avoid robotic or overly formal structures.
- Maintain a **friendly tone** in your answers, but prioritize **accuracy over formality**.
- Do not mention any file names or paths in your output.
- Form questions based on the subjects covered, then answer using the information found in the document.
- Where relevant, especially if the topic relates to **Houdini scripting**, include a **VEX code snippet** in your answer.
- Ensure that all code snippets are accurate, well-commented, and reflect best practices.
- Avoid speculation — only use information present in the source material.
- **Answer Length**: Provide comprehensive answers that fully address the question. Include all necessary details, steps, and explanations, but avoid unnecessary repetition or filler content. If a topic requires a detailed explanation with code examples, provide it. If a simple answer suffices, keep it concise.

#### Output Format:
{{
  "doc_chunk": "{content}",
  "qna": {{
    "question": "[Your generated human-style question]",
    "answer": "[Your detailed, friendly yet accurate answer - as long as needed to be complete]",
    "source": "houdini_docs"
  }}
}}

{{
  "doc_chunk": "{content}",
  "qna": {{
    "question": "[Second question]",
    "answer": "[Second answer]",
    "source": "houdini_docs"
  }}
}}

[Continue for 3-5 Q&A pairs]

Documentation: {content}

Generate the Q&A pairs in the specified JSON format:""",
    
    'odforce_forum': """You are tasked with generating **question-and-answer pairs** based on the content of a provided forum discussion.

#### General Instructions:
- Generate **2-4 natural language questions** and their corresponding **answers**.
- Extract the **main question** posed by the original poster, and summarize the **best/most helpful answer** given in the thread.
- Generate additional related questions from the context if multiple solutions or topics are discussed.
- Use **human-style phrasing** for questions — avoid robotic or overly formal structures.
- Maintain a **friendly tone** in your answers, but prioritize **accuracy over formality**.
- Do not mention any file names, usernames, or forum paths in your output.
- Where relevant, especially if the topic relates to **Houdini scripting**, include a **VEX code snippet** in your answer.
- Ensure that all code snippets are accurate, well-commented, and reflect best practices.
- Only generate questions that are actually answered in the discussion.
- Avoid speculation — only use information present in the source material.
- **Answer Length**: Provide comprehensive answers that capture the full solution or explanation from the forum discussion. Include all relevant details, alternative approaches, and code examples mentioned. Don't artificially shorten answers if the complete solution requires more explanation.

#### Output Format:
{{
  "doc_chunk": "{content}",
  "qna": {{
    "question": "[Your generated human-style question]",
    "answer": "[Your detailed, friendly yet accurate answer - as comprehensive as needed]",
    "source": "odforce_forum"
  }}
}}

{{
  "doc_chunk": "{content}",
  "qna": {{
    "question": "[Second question]",
    "answer": "[Second answer]",
    "source": "odforce_forum"
  }}
}}

[Continue for 2-4 Q&A pairs]

Discussion: {content}

Generate the Q&A pairs in the specified JSON format:""",
    
    'generic': """Generate 2-3 question-answer pairs from the following content. Use natural language and focus on the main topics covered. Provide complete, helpful answers that fully address each question without unnecessary verbosity.

#### Output Format:
{{
  "doc_chunk": "{content}",
  "qna": {{
    "question": "[Your generated question]",
    "answer": "[Your answer]",
    "source": "generic"
  }}
}}

Content: {content}

Generate the Q&A pairs in the specified JSON format:"""
}

def parse_markdown_to_plain_text(markdown_content: str) -> str:
    """Parse markdown content to plain text while preserving structure"""
    lines = markdown_content.split('\n')
    plain_text_lines = []
    for line in lines:
        # Remove markdown syntax (simplified)
        line = re.sub(r'#+\s*', '', line)  # Remove headers
        line = re.sub(r'\*\*(.+?)\*\*', r'\1', line)  # Remove bold
        line = re.sub(r'\*(.+?)\*', r'\1', line)  # Remove italic
        line = re.sub(r'`(.+?)`', r'\1', line)  # Remove code blocks
        line = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', line)  # Remove links
        plain_text_lines.append(line.strip())
    return '\n'.join(plain_text_lines)

def chunk_content(content: str, max_tokens: int = 2048) -> List[str]:
    """Split content into manageable chunks for processing"""
    words = content.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = ' '.join(words[i:i+max_tokens])
        if len(chunk.strip()) > 100:  # Only include substantial chunks
            chunks.append(chunk)
    return chunks

def preprocess_content(file_path: str) -> List[Dict]:
    """Preprocess content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    if len(content.strip()) < 100:  # Skip very short files
        return []

    plain_text = parse_markdown_to_plain_text(content)
    chunks = chunk_content(plain_text)

    return [{'content': chunk, 'metadata': {'source': file_path}} for chunk in chunks]

def measure_batch_performance(func):
    """Decorator to measure batch processing performance"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        total_time = end_time - start_time
        if hasattr(result, '__len__'):
            throughput = len(result) / total_time if total_time > 0 else 0
            print(f"Batch processing completed:")
            print(f"  - Total items: {len(result)}")
            print(f"  - Total time: {total_time:.2f}s")
            print(f"  - Throughput: {throughput:.2f} items/second")
        
        return result
    return wrapper

@measure_batch_performance
def batch_generate_qna_pairs_lm_studio(chunks_with_metadata: List[Dict], prompt_template: str) -> List[Dict]:
    """Generate QnA pairs using LM Studio API with batch processing"""
    if not chunks_with_metadata or not clients:
        return []
    
    print(f"Processing {len(chunks_with_metadata)} chunks with {len(clients)} LM Studio clients...")
    
    all_qna_pairs = []
    total_batches = (len(chunks_with_metadata) + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Process in batches (API calls are sequential but we batch the processing)
    for i in range(0, len(chunks_with_metadata), BATCH_SIZE):
        batch_chunks = chunks_with_metadata[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)")
        
        batch_start_time = time.time()
        batch_qna_pairs = []
        
        for j, chunk_data in enumerate(batch_chunks):
            try:
                content = chunk_data['content'][:2000]  # Limit content length
                prompt = prompt_template.format(content=content)
                
                # Calculate adaptive max tokens based on content complexity
                adaptive_max_tokens = calculate_adaptive_max_tokens(content)
                
                # Select client in round-robin fashion
                client_names = list(clients.keys())
                selected_client_name = client_names[j % len(client_names)]
                selected_client = clients[selected_client_name]['client']
                
                # Make API call to LM Studio with adaptive token limit
                response = selected_client.chat.completions.create(
                    model=selected_client_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=adaptive_max_tokens,
                    stop=None
                )
                
                response_text = response.choices[0].message.content.strip()
                qna_pairs = parse_qna_response(response_text)
                
                # Add metadata to each QnA pair including which model was used and token info
                for pair in qna_pairs:
                    pair['metadata'] = chunk_data['metadata'].copy()
                    pair['metadata']['model_used'] = selected_client_name
                    pair['metadata']['adaptive_max_tokens'] = adaptive_max_tokens
                
                    # If doc_chunk is empty (fallback parsing), use the original content
                    if not pair.get('doc_chunk'):
                        pair['doc_chunk'] = content
                
                batch_qna_pairs.extend(qna_pairs)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error processing chunk {j+1} in batch {batch_num}: {e}")
                continue
        
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        print(f"  Batch {batch_num} completed in {batch_time:.2f}s")
        print(f"  Generated {len(batch_qna_pairs)} Q&A pairs")
        
        all_qna_pairs.extend(batch_qna_pairs)
    
    print(f"Total QnA pairs generated: {len(all_qna_pairs)}")
    return all_qna_pairs

def parse_qna_response(response: str) -> List[Dict]:
    """Parse QnA pairs from model response in the new JSON format"""
    qna_pairs = []
    
    try:
        # Try to parse as JSON first (new format)
        import json
        
        # Clean up the response - remove any markdown code blocks
        cleaned_response = response.strip()
        if cleaned_response.startswith('```'):
            # Remove markdown code block markers
            lines = cleaned_response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            cleaned_response = '\n'.join(lines)
        
        # Try to parse multiple JSON objects
        # Split by lines and look for JSON objects
        json_objects = []
        current_json = ""
        brace_count = 0
        in_json = False
        
        for line in cleaned_response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('{'):
                if not in_json:
                    current_json = line
                    in_json = True
                    brace_count = line.count('{') - line.count('}')
                else:
                    current_json += '\n' + line
                    brace_count += line.count('{') - line.count('}')
            elif in_json:
                current_json += '\n' + line
                brace_count += line.count('{') - line.count('}')
                
                if brace_count == 0:
                    # Complete JSON object
                    try:
                        json_obj = json.loads(current_json)
                        json_objects.append(json_obj)
                    except json.JSONDecodeError:
                        pass
                    current_json = ""
                    in_json = False
        
        # Process parsed JSON objects
        for json_obj in json_objects:
            if isinstance(json_obj, dict) and 'doc_chunk' in json_obj and 'qna' in json_obj:
                qna_data = json_obj['qna']
                if 'question' in qna_data and 'answer' in qna_data:
                    qna_pairs.append({
                        'question': qna_data['question'].strip(),
                        'answer': qna_data['answer'].strip(),
                        'doc_chunk': json_obj['doc_chunk'].strip(),
                        'source_type': qna_data.get('source', 'unknown')
                    })
        
        if qna_pairs:
            return qna_pairs
            
    except Exception as e:
        print(f"JSON parsing failed: {e}, falling back to text parsing")
    
    # Fallback to old Q&A format parsing
    qa_pattern = r'(?:Q:|Question:)\s*(.+?)\s*(?:A:|Answer:)\s*(.+?)(?=(?:Q:|Question:)|$)'
    matches = re.findall(qa_pattern, response, re.DOTALL | re.IGNORECASE)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        
        # Clean up the text
        question = re.sub(r'\n+', ' ', question)
        answer = re.sub(r'\n+', ' ', answer)
        
        if len(question) > 10 and len(answer) > 20:
            qna_pairs.append({
                'question': question,
                'answer': answer,
                'doc_chunk': '',  # No context available in old format
                'source_type': 'unknown'
            })
    
    # If no Q&A format found, try alternative parsing
    if not qna_pairs:
        lines = response.split('\n')
        current_q = None
        current_a = None
        
        for line in lines:
            line = line.strip()
            if line.startswith(('Q:', 'Question:')):
                if current_q and current_a:
                    qna_pairs.append({
                        'question': current_q, 
                        'answer': current_a,
                        'doc_chunk': '',  # No context available in old format
                        'source_type': 'unknown'
                    })
                current_q = re.sub(r'^(Q:|Question:)\s*', '', line)
                current_a = None
            elif line.startswith(('A:', 'Answer:')):
                current_a = re.sub(r'^(A:|Answer:)\s*', '', line)
            elif current_a and line:
                current_a += ' ' + line
        
        # Add the last pair
        if current_q and current_a:
            qna_pairs.append({
                'question': current_q, 
                'answer': current_a,
                'doc_chunk': '',  # No context available in old format
                'source_type': 'unknown'
            })
    
    return qna_pairs

def validate_qna_pair(qna: Dict) -> bool:
    """Validate QnA pair with flexible length requirements"""
    question = qna['question'].strip()
    answer = qna['answer'].strip()
    
    # Basic length requirements
    if len(question) < 10 or len(answer) < 20:
        return False
    
    # Maximum length limits (more generous for answers)
    if len(question) > 400 or len(answer) > 3000:
        return False
    
    # Ensure it's actually a question
    if '?' not in question:
        return False
    
    # Quality checks - avoid overly repetitive or low-quality content
    words_in_answer = answer.split()
    if len(words_in_answer) < 10:  # Too short
        return False
    
    # Check for excessive repetition (simple heuristic)
    unique_words = set(words_in_answer)
    if len(unique_words) < len(words_in_answer) * 0.3:  # Less than 30% unique words
        return False
    
    # Validate doc_chunk exists (should always be present now)
    if not qna.get('doc_chunk'):
        return False
    
    return True

def collect_all_md_files(directory: str) -> List[str]:
    """Collect all markdown files from directory recursively"""
    md_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

@measure_batch_performance
def process_directory(directory: str):
    """Process all files in a directory with LM Studio API"""
    print(f"Collecting all .md files from {directory}...")
    md_files = collect_all_md_files(directory)
    print(f"Found {len(md_files)} markdown files")
    
    if not md_files:
        return []
    
    # Enhanced template selection based on directory
    if 'houdini' in directory.lower():
        prompt_template_key = 'houdini_docs'
        source_type = 'houdini'
    elif 'odforce' in directory.lower():
        prompt_template_key = 'odforce_forum'
        source_type = 'odforce'
    else:
        prompt_template_key = 'generic'
        source_type = 'unknown'
    
    print(f"Using prompt template: {prompt_template_key}")
    
    # Process files in batches to manage memory
    file_batch_size = 50  # Process 50 files at a time
    batch_paths = []  # Track saved batch files
    
    for i in range(0, len(md_files), file_batch_size):
        batch_files = md_files[i:i+file_batch_size]
        batch_num = i//file_batch_size + 1
        total_batches = (len(md_files) + file_batch_size - 1)//file_batch_size
        
        print(f"Processing file batch {batch_num}/{total_batches}")
        
        # Collect all chunks from this batch
        batch_chunks = []
        for file_path in tqdm(batch_files, desc="Preprocessing files"):
            chunks = preprocess_content(file_path)
            batch_chunks.extend(chunks)
        
        if batch_chunks:
            # Generate QnA pairs for this batch using LM Studio API
            if USE_CONCURRENT_PROCESSING and len(clients) > 1:
                batch_qna = batch_generate_qna_pairs_concurrent(batch_chunks, PROMPT_TEMPLATES[prompt_template_key])
            else:
                batch_qna = batch_generate_qna_pairs_lm_studio(batch_chunks, PROMPT_TEMPLATES[prompt_template_key])
            
            # Validate QnA pairs for this batch
            validated_batch_qna = [qna for qna in batch_qna if validate_qna_pair(qna)]
            
            print(f"Generated {len(validated_batch_qna)} valid QnA pairs from file batch {batch_num}")
            
            # Save this batch immediately
            if validated_batch_qna:
                batch_id = f"{batch_num:03d}"  # Zero-padded batch number
                batch_path = save_batch_dataset(validated_batch_qna, batch_id, source_type)
                if batch_path:
                    batch_paths.append(batch_path)
    
    print(f"Completed processing {directory}: {len(batch_paths)} batches saved")
    return batch_paths  # Return list of batch file paths instead of QnA pairs

def create_dataset_structure(qna_pairs: List[Dict]):
    """Create dataset structure with QnA pairs"""
    dataset = []
    for qna in qna_pairs:
        model_used = qna.get('metadata', {}).get('model_used', 'unknown')
        adaptive_tokens = qna.get('metadata', {}).get('adaptive_max_tokens', MAX_NEW_TOKENS)
        
        dataset.append({
            'doc_chunk': qna.get('doc_chunk', ''),
            'qna': {
                'question': qna['question'],
                'answer': qna['answer'],
                'source': qna.get('metadata', {}).get('source', 'unknown')
            },
            'source_document': qna.get('metadata', {}).get('source', 'unknown'),
            'topic_hierarchy': extract_topic_hierarchy(qna.get('metadata', {}).get('source', '')),
            'generated_metadata': {
                'model': model_used,
                'generation_params': {
                    'temperature': 0.7,
                    'base_max_tokens': MAX_NEW_TOKENS,
                    'adaptive_max_tokens': adaptive_tokens,
                    'framework': 'LM Studio API',
                    'concurrent_processing': USE_CONCURRENT_PROCESSING,
                    'total_models_used': len(clients),
                    'adaptive_token_strategy': 'content_complexity_based'
                }
            },
            'confidence_score': 1.0,  # Placeholder
            'related_questions': []  # Placeholder
        })
    return dataset

def extract_topic_hierarchy(file_path: str) -> List[str]:
    """Extract topic hierarchy from file path"""
    if not file_path:
        return []
    
    # Remove the base directory and file extension
    relative_path = file_path
    for base_dir in [HOUDINI_DOCS_MKDOWN, ODFORCE_SCRAPMD]:
        if base_dir in relative_path:
            relative_path = relative_path.split(base_dir)[-1]
            break
    
    # Split path into hierarchy
    hierarchy = [part for part in relative_path.split(os.sep) if part and part != '.md']
    return hierarchy

def export_dataset(dataset: List[Dict], output_format: str = 'json'):
    """Export dataset to specified format"""
    if not os.path.exists(QNA_DATASET_DIR):
        os.makedirs(QNA_DATASET_DIR)

    if output_format == 'json':
        output_path = os.path.join(QNA_DATASET_DIR, 'complete_dataset.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        print(f"Dataset exported to {output_path}")
    elif output_format == 'csv':
        output_path = os.path.join(QNA_DATASET_DIR, 'complete_dataset.csv')
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            if dataset:
                # Flatten nested fields for CSV
                flattened_dataset = []
                for item in dataset:
                    flat_item = {
                        'doc_chunk': item['doc_chunk'],
                        'question': item['qna']['question'],
                        'answer': item['qna']['answer'],
                        'qna_source': item['qna']['source'],
                        'source_document': item['source_document'],
                        'topic_hierarchy': ' > '.join(item['topic_hierarchy']),
                        'model': item['generated_metadata']['model'],
                        'framework': item['generated_metadata']['generation_params']['framework'],
                        'confidence_score': item['confidence_score']
                    }
                    flattened_dataset.append(flat_item)
                
                writer = csv.DictWriter(f, fieldnames=flattened_dataset[0].keys())
                writer.writeheader()
                writer.writerows(flattened_dataset)
        print(f"Dataset exported to {output_path}")
    elif output_format == 'md':
        output_path = os.path.join(QNA_DATASET_DIR, 'complete_dataset.md')
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(f"## {item['qna']['question']}\n\n")
                f.write(f"**Context:**\n```\n{item['doc_chunk']}\n```\n\n")
                f.write(f"{item['qna']['answer']}\n\n")
                f.write(f"**Source:** {item['source_document']}\n\n")
                f.write(f"**Topic:** {' > '.join(item['topic_hierarchy'])}\n\n")
                f.write("---\n\n")
        print(f"Dataset exported to {output_path}")

def save_batch_dataset(qna_pairs: List[Dict], batch_id: str, source_type: str):
    """Save a batch of QnA pairs to individual file"""
    if not qna_pairs:
        return None
        
    # Create batch directory if it doesn't exist
    batch_dir = os.path.join(QNA_DATASET_DIR, 'batches')
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    
    # Create dataset structure for this batch
    batch_dataset = create_dataset_structure(qna_pairs)
    
    # Save batch as JSON
    batch_filename = f"{source_type}_batch_{batch_id}.json"
    batch_path = os.path.join(batch_dir, batch_filename)
    
    with open(batch_path, 'w', encoding='utf-8') as f:
        json.dump(batch_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Batch saved: {batch_path} ({len(batch_dataset)} Q&A pairs)")
    return batch_path

def consolidate_batch_datasets():
    """Consolidate all batch datasets into final complete dataset"""
    batch_dir = os.path.join(QNA_DATASET_DIR, 'batches')
    
    if not os.path.exists(batch_dir):
        print("No batch directory found. Nothing to consolidate.")
        return []
    
    # Collect all batch files
    batch_files = [f for f in os.listdir(batch_dir) if f.endswith('.json')]
    
    if not batch_files:
        print("No batch files found. Nothing to consolidate.")
        return []
    
    print(f"Consolidating {len(batch_files)} batch files...")
    
    consolidated_dataset = []
    batch_stats = {'houdini': 0, 'odforce': 0}
    
    for batch_file in sorted(batch_files):
        batch_path = os.path.join(batch_dir, batch_file)
        try:
            with open(batch_path, 'r', encoding='utf-8') as f:
                batch_data = json.load(f)
                consolidated_dataset.extend(batch_data)
                
                # Track statistics
                if 'houdini' in batch_file:
                    batch_stats['houdini'] += len(batch_data)
                elif 'odforce' in batch_file:
                    batch_stats['odforce'] += len(batch_data)
                
                print(f"Loaded {len(batch_data)} Q&A pairs from {batch_file}")
        except Exception as e:
            print(f"Error loading batch file {batch_file}: {e}")
    
    print(f"Consolidation complete: {len(consolidated_dataset)} total Q&A pairs")
    print(f"  - Houdini docs: {batch_stats['houdini']} pairs")
    print(f"  - ODForce forum: {batch_stats['odforce']} pairs")
    
    return consolidated_dataset

def process_chunk_with_model(chunk_data: Dict, prompt_template: str, model_info: Tuple[str, Dict]) -> List[Dict]:
    """Process a single chunk with a specific model"""
    model_name, model_data = model_info
    client = model_data['client']
    
    try:
        content = chunk_data['content'][:2000]  # Limit content length
        prompt = prompt_template.format(content=content)
        
        # Calculate adaptive max tokens based on content complexity
        adaptive_max_tokens = calculate_adaptive_max_tokens(content)
        
        # Make API call to LM Studio with adaptive token limit
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=adaptive_max_tokens,
            stop=None
        )
        
        response_text = response.choices[0].message.content.strip()
        qna_pairs = parse_qna_response(response_text)
        
        # Add metadata to each QnA pair including which model was used and token info
        for pair in qna_pairs:
            pair['metadata'] = chunk_data['metadata'].copy()
            pair['metadata']['model_used'] = model_name
            pair['metadata']['adaptive_max_tokens'] = adaptive_max_tokens
            
            # If doc_chunk is empty (fallback parsing), use the original content
            if not pair.get('doc_chunk'):
                pair['doc_chunk'] = content
        
        return qna_pairs
        
    except Exception as e:
        print(f"Error processing chunk with model {model_name}: {e}")
        return []

@measure_batch_performance
def batch_generate_qna_pairs_concurrent(chunks_with_metadata: List[Dict], prompt_template: str) -> List[Dict]:
    """Generate QnA pairs using multiple LM Studio clients concurrently"""
    if not chunks_with_metadata or not clients:
        return []
    
    print(f"Processing {len(chunks_with_metadata)} chunks with {len(clients)} concurrent LM Studio clients...")
    
    all_qna_pairs = []
    
    # Use ThreadPoolExecutor for concurrent processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a list of (chunk, model) pairs for balanced distribution
        chunk_model_pairs = []
        client_items = list(clients.items())
        
        for i, chunk_data in enumerate(chunks_with_metadata):
            # Distribute chunks across models in round-robin fashion
            model_info = client_items[i % len(client_items)]
            chunk_model_pairs.append((chunk_data, prompt_template, model_info))
        
        # Submit all tasks
        future_to_chunk = {
            executor.submit(process_chunk_with_model, chunk_data, prompt_template, model_info): i
            for i, (chunk_data, prompt_template, model_info) in enumerate(chunk_model_pairs)
        }
        
        # Collect results with progress bar
        completed = 0
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_index = future_to_chunk[future]
            try:
                qna_pairs = future.result()
                all_qna_pairs.extend(qna_pairs)
                completed += 1
                
                if completed % 10 == 0:  # Progress update every 10 completions
                    print(f"  Completed {completed}/{len(chunks_with_metadata)} chunks")
                    
            except Exception as e:
                print(f"Chunk {chunk_index} generated an exception: {e}")
    
    print(f"Concurrent processing completed: {len(all_qna_pairs)} Q&A pairs generated")
    return all_qna_pairs

def calculate_adaptive_max_tokens(content: str, base_max_tokens: int = MAX_NEW_TOKENS) -> int:
    """Calculate adaptive max tokens based on content complexity"""
    content_length = len(content)
    
    # Count technical indicators that might require longer explanations
    technical_indicators = [
        'vex', 'code', 'function', 'parameter', 'attribute', 'node', 'shader',
        'expression', 'script', 'python', 'hscript', 'workflow', 'tutorial',
        'example', 'step', 'procedure', 'algorithm', 'implementation'
    ]
    
    technical_count = sum(1 for indicator in technical_indicators 
                         if indicator.lower() in content.lower())
    
    # Base calculation
    if content_length < 500:
        # Short content - likely needs concise answers
        adaptive_tokens = int(base_max_tokens * 0.6)
    elif content_length < 1500:
        # Medium content - standard tokens
        adaptive_tokens = base_max_tokens
    else:
        # Long content - might need detailed explanations
        adaptive_tokens = int(base_max_tokens * 1.3)
    
    # Adjust based on technical complexity
    if technical_count > 5:
        # High technical content - likely needs code examples and detailed explanations
        adaptive_tokens = int(adaptive_tokens * 1.4)
    elif technical_count > 2:
        # Moderate technical content
        adaptive_tokens = int(adaptive_tokens * 1.2)
    
    # Ensure reasonable bounds
    min_tokens = 256
    max_tokens = 2048  # Reasonable upper limit
    
    return max(min_tokens, min(adaptive_tokens, max_tokens))

def main():
    # Setup LM Studio clients
    if not setup_lm_studio_clients():
        print("Failed to setup LM Studio clients. Exiting.")
        print("\nPlease ensure:")
        print("1. LM Studio instance is running on the specified port")
        print("2. API server is enabled in LM Studio")
        print("3. Models are loaded in the instance")
        print("4. Model names match exactly (including any :2, :3 suffixes)")
        return

    print("\n=== Starting LM Studio API QnA Generation ===")
    print(f"Concurrent processing: {'Enabled' if USE_CONCURRENT_PROCESSING else 'Disabled'}")
    print(f"Available models: {list(clients.keys())}")
    print(f"Max concurrent workers: {MAX_WORKERS}")
    print(f"Base max tokens: {MAX_NEW_TOKENS}")
    print(f"Adaptive token range: 256-2048 (based on content complexity)")
    print(f"Token adaptation: Content length + technical complexity analysis")
    start_time = time.time()

    # Process all directories and save batches individually
    print("Processing ALL Houdini documentation files...")
    houdini_batch_paths = process_directory(HOUDINI_DOCS_MKDOWN)
    
    print("Processing ALL ODForce forum files...")
    odforce_batch_paths = process_directory(ODFORCE_SCRAPMD)

    # Check if any batches were created
    total_batches = len(houdini_batch_paths) + len(odforce_batch_paths)
    if total_batches == 0:
        print("No batches were generated. Please check the input data and LM Studio configuration.")
        return

    print(f"\n=== Batch Processing Summary ===")
    print(f"Houdini documentation batches: {len(houdini_batch_paths)}")
    print(f"ODForce forum batches: {len(odforce_batch_paths)}")
    print(f"Total batches saved: {total_batches}")

    # Consolidate all batches into final dataset
    print("\n=== Consolidating Batches ===")
    consolidated_dataset = consolidate_batch_datasets()

    if consolidated_dataset:
        # Export consolidated datasets in all formats
        print("\n=== Exporting Final Dataset ===")
        export_dataset(consolidated_dataset, 'json')
        export_dataset(consolidated_dataset, 'csv')
        export_dataset(consolidated_dataset, 'md')
        
        # Print summary statistics
        end_time = time.time()
        total_time = end_time - start_time
        
        print("\n=== Final Dataset Summary ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Total Q&A pairs: {len(consolidated_dataset)}")
        
        # Calculate throughput
        throughput = len(consolidated_dataset) / total_time if total_time > 0 else 0
        print(f"Overall throughput: {throughput:.2f} Q&A pairs/second")
        
        # Count by source type
        houdini_count = sum(1 for item in consolidated_dataset if 'houdini' in item.get('source_document', '').lower())
        odforce_count = len(consolidated_dataset) - houdini_count
        print(f"Houdini docs pairs: {houdini_count}")
        print(f"ODForce forum pairs: {odforce_count}")
        
        # Count by topic
        topic_counts = {}
        model_counts = {}
        token_stats = {'min': float('inf'), 'max': 0, 'total': 0, 'count': 0}
        
        for item in consolidated_dataset:
            if item['topic_hierarchy']:
                main_topic = item['topic_hierarchy'][0]
                topic_counts[main_topic] = topic_counts.get(main_topic, 0) + 1
            
            # Count by model used
            model_used = item['generated_metadata']['model']
            model_counts[model_used] = model_counts.get(model_used, 0) + 1
            
            # Track adaptive token usage
            adaptive_tokens = item['generated_metadata']['generation_params'].get('adaptive_max_tokens', MAX_NEW_TOKENS)
            token_stats['min'] = min(token_stats['min'], adaptive_tokens)
            token_stats['max'] = max(token_stats['max'], adaptive_tokens)
            token_stats['total'] += adaptive_tokens
            token_stats['count'] += 1
        
        print("\nTop topics by Q&A count:")
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {topic}: {count}")
        
        print("\nQ&A pairs by model:")
        for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {model}: {count}")
        
        # Adaptive token statistics
        if token_stats['count'] > 0:
            avg_tokens = token_stats['total'] / token_stats['count']
            print(f"\nAdaptive Token Usage Statistics:")
            print(f"  Average tokens per response: {avg_tokens:.1f}")
            print(f"  Token range used: {token_stats['min']}-{token_stats['max']}")
            print(f"  Base max tokens: {MAX_NEW_TOKENS}")
            efficiency = (avg_tokens / MAX_NEW_TOKENS) * 100
            print(f"  Token efficiency: {efficiency:.1f}% of base limit")
        
        print("\nLM Studio API dataset generation completed successfully!")
        print(f"Individual batches saved in: {os.path.join(QNA_DATASET_DIR, 'batches')}")
        print(f"Final consolidated dataset saved in: {QNA_DATASET_DIR}")
        
        # Performance comparison note
        print(f"\n=== Performance Notes ===")
        print(f"Framework: LM Studio API (OpenAI-compatible)")
        print(f"Concurrent processing: {'Enabled' if USE_CONCURRENT_PROCESSING else 'Disabled'}")
        print(f"Models used: {len(clients)}")
        print(f"GPU: RTX 5070 Ti with full sm_120 support")
        if USE_CONCURRENT_PROCESSING and len(clients) > 1:
            print(f"Expected speedup: ~{len(clients)}x faster than single model")
        else:
            print(f"Running in sequential mode")
        
    else:
        print("No consolidated dataset was created. Please check the batch files.")

if __name__ == '__main__':
    main()
