#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QnA Dataset Generation using Unsloth
This script generates question-answer pairs from Houdini documentation and ODForce forum posts.
"""

import time
import os
import pandas as pd
from pathlib import Path
from datasets import Dataset
from unsloth.dataprep import SyntheticDataKit

def init_generator(model_name="unsloth/Qwen3-4B-unsloth-bnb-4bit", max_seq_length=2048):
    """Initialize the synthetic data generator."""
    generator = SyntheticDataKit.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        system_prompt="/nothink"  # Disable thinking mode for Qwen 3 4B
    )
    
    generator.prepare_qa_generation(
        output_folder="data",
        temperature=0.7,
        top_p=0.95,
        overlap=64,
        max_generation_tokens=512,
    )
    
    return generator

def collect_markdown_files(base_dirs):
    """Collect all markdown files from the specified directories."""
    markdown_files = []
    
    for base_dir in base_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith('.md'):
                    markdown_files.append(os.path.join(root, file))
    
    return markdown_files

def process_markdown_file(generator, filepath):
    """Process a single markdown file and generate chunks."""
    # Create output directory if it doesn't exist
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a safe filename for the output
    safe_filename = Path(filepath).stem.replace(" ", "_")
    output_file = output_dir / f"{safe_filename}.txt"
    
    # Read and write the markdown content
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Chunk the file
    return generator.chunk_data(str(output_file))

def process_chunks(generator, filenames, num_pairs_per_chunk=25):
    """Process document chunks to generate QA pairs."""
    for filename in filenames:
        # Generate QA pairs for each chunk
        generator.run_command(
            f"synthetic-data-kit -c synthetic_data_kit_config.yaml create {filename} --num-pairs {num_pairs_per_chunk} --type 'qa'"
        )
        time.sleep(2)  # Allow processing time

def convert_to_qa_format(filenames):
    """Convert generated datasets into QA format."""
    for i, filename in enumerate(filenames):
        base_name = Path(filename).stem
        qa_pairs_filename = f"data/generated/{base_name}_qa_pairs.json"
        
        generator.run_command(
            f"synthetic-data-kit -c synthetic_data_kit_config.yaml save-as {qa_pairs_filename} -f ft"
        )

def load_dataset(filenames):
    """Load and combine the generated QA pairs into a dataset."""
    final_filenames = [
        f"data/final/{Path(filename).stem}_qa_pairs_ft.json"
        for filename in filenames
    ]
    
    # Load all available JSON files
    dataframes = []
    for name in final_filenames:
        try:
            df = pd.read_json(name)
            dataframes.append(df)
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            print(f"Warning: Could not load {name}: {e}")
    
    if not dataframes:
        raise ValueError("No valid datasets found to combine")
    
    conversations = pd.concat(dataframes).reset_index(drop=True)
    return Dataset.from_pandas(conversations)

def main():
    # Source directories
    source_dirs = [
        "houdini_docs_mkdown",
        "odforce_scrapMD"
    ]
    
    # Initialize generator
    generator = init_generator()
    
    # Collect all markdown files
    print("Collecting markdown files...")
    markdown_files = collect_markdown_files(source_dirs)
    print(f"Found {len(markdown_files)} markdown files")
    
    # Process each markdown file
    all_chunks = []
    for md_file in markdown_files:
        print(f"Processing {md_file}...")
        chunks = process_markdown_file(generator, md_file)
        all_chunks.extend(chunks)
        print(f"Generated {len(chunks)} chunks")
    
    # Process chunks to generate QA pairs
    print("Generating QA pairs...")
    process_chunks(generator, all_chunks)
    
    # Convert to QA format
    print("Converting to QA format...")
    convert_to_qa_format(all_chunks)
    
    # Load and combine datasets
    print("Loading combined dataset...")
    dataset = load_dataset(all_chunks)
    print(f"Total QA pairs: {len(dataset)}")
    
    # Save the combined dataset
    output_dir = Path("data/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(output_dir / "combined_dataset")
    print(f"Dataset saved to {output_dir}/combined_dataset")
    
    # Cleanup
    print("Cleaning up...")
    generator.cleanup()

if __name__ == "__main__":
    main()