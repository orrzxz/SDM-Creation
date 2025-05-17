import os
import json
import google.generativeai as genai
import time
import sys

# --- Configuration ---
API_KEY = "AIzaSyDTj37d7HBpfv3dGoE-odaB_ZKQDhTE9BQ" # Replace with your actual API key if needed
INPUT_DIR = "PDF_to_Markdown"
OUTPUT_DIR = "Chunked_Markdown"
MODEL_NAME = "gemini-2.5-pro-preview-03-25"
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds

# --- Safety Settings for Gemini ---
# Adjust these as needed, be cautious with HARM_BLOCK_THRESHOLD_NONE
# safety_settings = {
#     "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
#     "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
#     "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
#     "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
# }

# --- Prompt for Semantic Chunking ---
# This prompt asks the model to return a JSON list of strings.
# You might need to refine this prompt based on the results you get.
CHUNK_PROMPT = """
Analyze the following Markdown text from a technical documentation page. Divide the text into semantically meaningful chunks. Each chunk should represent a distinct topic, concept, or section. Preserve the original Markdown formatting within each chunk where appropriate.

Return the result ONLY as a valid JSON list of strings, where each string is a chunk. Do not include any introductory text, explanations, or markdown formatting around the JSON list itself.

Example Input Markdown:
# Section 1
This is the first paragraph.

## Subsection 1.1
Details about subsection 1.1.

# Section 2
This section covers a different topic.
- Point 1
- Point 2

Example Output JSON:
[
  "# Section 1\\nThis is the first paragraph.",
  "## Subsection 1.1\\nDetails about subsection 1.1.",
  "# Section 2\\nThis section covers a different topic.\\n- Point 1\\n- Point 2"
]

Now, process the following Markdown text:

--- START OF TEXT ---
{markdown_text}
--- END OF TEXT ---
"""

# --- API Interaction ---
def chunk_text_with_gemini(text, model):
    """Sends text to Gemini API for chunking and returns chunks."""
    prompt = CHUNK_PROMPT.format(markdown_text=text)
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = model.generate_content(
                prompt,
                # safety_settings=safety_settings,
                generation_config=genai.types.GenerationConfig(
                    # Ensure JSON output if supported, otherwise rely on prompt
                    # response_mime_type="application/json", # Uncomment if model/API supports explicit JSON mode
                    temperature=0.1 # Lower temperature for more deterministic chunking
                )
            )
            # Accessing response text safely
            if response.parts:
                 # Attempt to parse the JSON response
                try:
                    # Clean potential markdown code fences if model wraps JSON
                    cleaned_response = response.text.strip().strip('```json').strip('```')
                    chunks = json.loads(cleaned_response)
                    if isinstance(chunks, list) and all(isinstance(item, str) for item in chunks):
                        return chunks
                    else:
                        print(f"   [Warning] API returned valid JSON but not a list of strings: {type(chunks)}")
                        # Fallback: return the raw text as a single chunk if parsing fails structurally
                        return [response.text]
                except json.JSONDecodeError as json_err:
                    print(f"   [Error] Failed to decode JSON response: {json_err}")
                    print(f"   Raw Response Text: {response.text[:500]}...") # Print beginning of raw response
                    # Fallback: return the raw text as a single chunk
                    return [response.text]
                except Exception as e:
                    print(f"   [Error] Unexpected error processing response: {e}")
                    return [response.text] # Fallback
            elif response.prompt_feedback.block_reason:
                 print(f"   [Error] Request blocked. Reason: {response.prompt_feedback.block_reason}")
                 return None # Indicate blocking
            else:
                 print("   [Warning] Received empty response from API.")
                 return [] # Return empty list for empty response

        except Exception as e:
            retries += 1
            print(f"   [Error] API call failed: {e}. Retrying ({retries}/{MAX_RETRIES})...")
            if retries >= MAX_RETRIES:
                print(f"   [Error] Max retries reached. Skipping chunking for this part.")
                return None # Indicate failure after retries
            time.sleep(RETRY_DELAY)
    return None # Should not be reached if loop logic is correct

# --- Main Processing Logic ---
def process_files():
    """Finds markdown files, chunks them, and saves the results."""
    print(f"Starting processing...")
    print(f"Input directory: {os.path.abspath(INPUT_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    if not os.path.isdir(INPUT_DIR):
        print(f"[Error] Input directory '{INPUT_DIR}' not found.")
        sys.exit(1)

    # Configure Gemini API
    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini API configured with model: {MODEL_NAME}")
    except Exception as e:
        print(f"[Error] Failed to configure Gemini API: {e}")
        sys.exit(1)

    file_count = 0
    processed_count = 0
    error_count = 0

    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if filename.lower().endswith(".md"):
                file_count += 1
                input_filepath = os.path.join(root, filename)
                relative_path = os.path.relpath(input_filepath, INPUT_DIR)
                output_filename = os.path.splitext(filename)[0] + ".json"
                output_subdir = os.path.dirname(relative_path)
                output_path_dir = os.path.join(OUTPUT_DIR, output_subdir)
                output_filepath = os.path.join(output_path_dir, output_filename)

                print(f"\nProcessing file {file_count}: {relative_path}")

                # Create output directory if it doesn't exist
                os.makedirs(output_path_dir, exist_ok=True)

                try:
                    with open(input_filepath, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()

                    if not markdown_content.strip():
                        print("   [Info] File is empty. Skipping.")
                        continue

                    print(f"   Sending {len(markdown_content)} characters to Gemini API...")
                    chunks = chunk_text_with_gemini(markdown_content, model)

                    if chunks is not None:
                        print(f"   Received {len(chunks)} chunks. Saving to {output_filepath}")
                        with open(output_filepath, 'w', encoding='utf-8') as f:
                            json.dump(chunks, f, indent=2, ensure_ascii=False)
                        processed_count += 1
                    else:
                        print(f"   [Error] Failed to get chunks for {relative_path} after retries.")
                        error_count += 1

                except FileNotFoundError:
                    print(f"   [Error] File not found (should not happen in os.walk): {input_filepath}")
                    error_count += 1
                except Exception as e:
                    print(f"   [Error] Failed to process file {input_filepath}: {e}")
                    error_count += 1

    print(f"\n--- Processing Summary ---")
    print(f"Total Markdown files found: {file_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Errors/Skipped: {error_count}")
    print(f"Chunked files saved in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    process_files()
