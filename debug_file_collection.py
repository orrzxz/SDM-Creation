#!/usr/bin/env python3

import os
from typing import List

def collect_all_md_files(directory: str) -> List[str]:
    """Collect all markdown files from directory recursively"""
    print(f"Starting to collect files from: {directory}")
    print(f"Directory exists: {os.path.exists(directory)}")
    print(f"Directory is dir: {os.path.isdir(directory)}")
    
    md_files = []
    try:
        for root, dirs, files in os.walk(directory):
            print(f"Walking: {root}, found {len(files)} files")
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    md_files.append(full_path)
                    if len(md_files) <= 5:  # Print first 5 files found
                        print(f"  Found: {full_path}")
            if len(md_files) > 10:  # Stop after finding 10 files for testing
                break
    except Exception as e:
        print(f"Error during os.walk: {e}")
    
    return md_files

if __name__ == "__main__":
    print("Testing file collection...")
    
    # Test houdini docs
    print("\n=== Testing Houdini Docs ===")
    houdini_files = collect_all_md_files('houdini_docs_mkdown')
    print(f"Found {len(houdini_files)} files")
    
    # Test odforce
    print("\n=== Testing ODForce ===")
    odforce_files = collect_all_md_files('odforce_scrapMD')
    print(f"Found {len(odforce_files)} files") 