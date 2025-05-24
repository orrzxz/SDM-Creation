#!/usr/bin/env python3

import os
import json
from QnAGenerator_finalFinal_final import (
    setup_lm_studio_clients, 
    preprocess_content, 
    batch_generate_qna_pairs_lm_studio,
    PROMPT_TEMPLATES,
    parse_qna_response,
    create_dataset_structure
)

def test_new_format():
    """Test the new context-aware QnA format with a small sample"""
    
    # Setup clients
    print("Setting up LM Studio clients...")
    if not setup_lm_studio_clients():
        print("Failed to setup clients")
        return
    
    # Find a few test files
    test_files = []
    for root, dirs, files in os.walk('houdini_docs_mkdown'):
        for file in files[:3]:  # Just take first 3 files
            if file.endswith('.md'):
                test_files.append(os.path.join(root, file))
        if len(test_files) >= 3:
            break
    
    print(f"Testing with {len(test_files)} files:")
    for f in test_files:
        print(f"  - {f}")
    
    # Process files
    all_chunks = []
    for file_path in test_files:
        chunks = preprocess_content(file_path)
        all_chunks.extend(chunks[:1])  # Just one chunk per file
    
    print(f"Processing {len(all_chunks)} chunks...")
    
    # Generate QnA pairs
    qna_pairs = batch_generate_qna_pairs_lm_studio(all_chunks, PROMPT_TEMPLATES['houdini_docs'])
    
    print(f"Generated {len(qna_pairs)} QnA pairs")
    
    # Test the new format
    for i, qna in enumerate(qna_pairs[:2]):  # Show first 2
        print(f"\n=== QnA Pair {i+1} ===")
        print(f"Doc Chunk: {qna.get('doc_chunk', 'MISSING')[:100]}...")
        print(f"Question: {qna.get('question', 'MISSING')}")
        print(f"Answer: {qna.get('answer', 'MISSING')[:100]}...")
        print(f"Source Type: {qna.get('source_type', 'MISSING')}")
    
    # Test dataset structure
    dataset = create_dataset_structure(qna_pairs)
    
    print(f"\n=== Dataset Structure Test ===")
    if dataset:
        sample = dataset[0]
        print(f"Has doc_chunk: {'doc_chunk' in sample}")
        print(f"Has nested qna: {'qna' in sample}")
        if 'qna' in sample:
            print(f"QnA has question: {'question' in sample['qna']}")
            print(f"QnA has answer: {'answer' in sample['qna']}")
            print(f"QnA has source: {'source' in sample['qna']}")
    
    # Save test output
    with open('test_output.json', 'w') as f:
        json.dump(dataset[:2], f, indent=2)
    
    print(f"\nTest completed! Sample output saved to test_output.json")

if __name__ == '__main__':
    test_new_format() 