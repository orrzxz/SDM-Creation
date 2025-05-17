import os
import glob
import argparse
import json
import subprocess


def collect_markdown_files(input_dirs):
    """
    Recursively collect all .md files from given input directories.
    """
    md_files = []
    for root in input_dirs:
        pattern = os.path.join(root, '**', '*.md')
        md_files.extend(glob.glob(pattern, recursive=True))
    return md_files


def read_markdown(path):
    """
    Read markdown file content.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_processed_text(md_file, content, input_dirs, proc_dir):
    """
    Mirror the markdown file structure into processed_text and write the content.
    Chooses the first matching input_dir as base for relative path.
    """
    abs_path = os.path.abspath(md_file)
    base_dir = None
    for root in input_dirs:
        abs_root = os.path.abspath(root)
        if abs_path.startswith(abs_root + os.sep):
            base_dir = abs_root
            break
    rel_path = os.path.relpath(abs_path, base_dir) if base_dir else os.path.basename(md_file)
    out_path = os.path.join(proc_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return out_path


def generate_qna_from_docs(md_files, input_dirs, proc_dir, output_dir, samples, verbose=False):
    """
    Generate synthetic Q&A using synthetic-data-kit CLI.
    """
    # Ensure output dirs
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    all_qna = []
    for md_file in md_files:
        content = read_markdown(md_file)
        # Write processed text
        proc_file = write_processed_text(md_file, content, input_dirs, proc_dir)
        if verbose:
            print(f"Processed markdown: {proc_file}")

        # Generate synthetic Q&A pairs using synthetic-data-kit CLI
        output_json = os.path.join(output_dir, f"{os.path.basename(md_file)}_qa_pairs.json")
        
        try:
            # Create QA pairs
            subprocess.run([
                'synthetic-data-kit',
                'create',
                proc_file,
                '--type', 'qa',
                '--output', output_json,
                '--num-pairs', str(samples)
            ], check=True)

            # Read generated QA pairs
            with open(output_json, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
                for qa in qa_pairs:
                    qa['source_file'] = md_file
                all_qna.extend(qa_pairs)

        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Error generating QA pairs for {md_file}: {e}")
            continue

    # Write all QA pairs to final output file
    output_path = os.path.join(output_dir, 'synthetic_qna.jsonl')
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for qa in all_qna:
            out_f.write(json.dumps(qa, ensure_ascii=False) + '\n')

    if verbose:
        print(f"Wrote {len(all_qna)} Q&A pairs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A dataset from markdown folder structure into qna_dataset"
    )
    parser.add_argument(
        '--inputs', '-i', nargs='+', required=True,
        help='List of root directories to scan for markdown files (e.g., odforce_scrapMD houdini_docs_mkdown).'
    )
    parser.add_argument(
        '--base', '-b', default='qna_dataset',
        help='Base folder for processed_text and output subfolders.'
    )
    parser.add_argument(
        '--samples', '-s', type=int, default=100,
        help='Number of synthetic Q&A samples to generate per document.'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging.'
    )
    args = parser.parse_args()

    proc_dir = os.path.join(args.base, 'processed_text')
    output_dir = os.path.join(args.base, 'output')

    md_files = collect_markdown_files(args.inputs)
    if args.verbose:
        print(f"Found {len(md_files)} markdown files in {args.inputs}.")

    generate_qna_from_docs(
        md_files,
        args.inputs,
        proc_dir,
        output_dir,
        args.samples,
        args.verbose
    )

if __name__ == '__main__':
    main()
