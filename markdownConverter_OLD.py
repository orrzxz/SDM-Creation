import os
from markitdown import MarkItDown

root_folder = r'E:\!Python\SANDBOX\houdiniDocsCrawler\houdini_docs_pdf'
output_root_folder = r'E:\!Python\SANDBOX\houdiniDocsCrawler\PDF_to_Markdown'

def convert_pdfs_to_markdown(root_folder, output_root_folder):
    """
    Finds all PDF files in root_folder, converts them to Markdown,
    and saves them to output_root_folder, preserving the directory structure.
    """
    md = MarkItDown(enable_plugins=False) # Initialize converter once

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.pdf'):
                pdf_filepath = os.path.join(dirpath, filename)

                # Calculate relative path and construct output path
                relative_path = os.path.relpath(dirpath, root_folder)
                output_dir = os.path.join(output_root_folder, relative_path)
                base_filename = os.path.splitext(filename)[0]
                md_filename = f"{base_filename}.md"
                md_filepath = os.path.join(output_dir, md_filename)

                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)

                try:
                    print(f"Converting: {pdf_filepath}")
                    result = md.convert(pdf_filepath)
                    with open(md_filepath, 'w', encoding='utf-8') as f:
                        f.write(result.text_content)
                    print(f"Saved Markdown to: {md_filepath}")
                except Exception as e:
                    print(f"Error converting {pdf_filepath}: {e}")

# Start the conversion process
convert_pdfs_to_markdown(root_folder, output_root_folder)

print("Conversion process finished.")