import os
import requests
import pdfkit
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
import logging
import re

# --- Configuration ---
START_URL = "https://www.sidefx.com/docs/houdini/"
BASE_DOMAIN = urlparse(START_URL).netloc # e.g., 'www.sidefx.com'
BASE_PATH = urlparse(START_URL).path     # e.g., '/docs/houdini/'
OUTPUT_DIR = "houdini_docs_pdf"
REQUEST_DELAY = 0.5 # Seconds between requests to be polite to the server
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36" # Be a good citizen

# Configure wkhtmltopdf path (if not in system PATH) - CHANGE IF NEEDED
WKHTMLTOPDF_PATH = 'C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe' # Set to r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe' on Windows, or '/usr/local/bin/wkhtmltopdf' etc. if needed

PDFKIT_OPTIONS = {
    'page-size': 'A4',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': "UTF-8",
    'custom-header': [
        ('Accept-Encoding', 'gzip')
    ],
    'cookie': [], # Add cookies if login is needed (not required for sidefx docs usually)
    'outline': None,
    'enable-local-file-access': None, # May be needed if CSS/JS are local/complex
    'load-error-handling': 'ignore', # Ignore errors loading some resources (e.g., missing images)
    'load-media-error-handling': 'ignore',
    'javascript-delay': 2000, # Give JS 2 seconds to load/render, adjust if needed
    'no-stop-slow-scripts': None,
    'user-style-sheet': None, # Optional: path to a custom CSS file
    'quiet': '', # Suppress wkhtmltopdf output unless errors occur
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def sanitize_filename(filename):
    """Removes or replaces characters invalid for filenames."""
    # Remove characters not allowed in most file systems
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Replace potential multiple underscores with a single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores/spaces
    sanitized = sanitized.strip('_. ')
    # Limit length if necessary (e.g., 200 chars) - uncomment if needed
    # max_len = 200
    # if len(sanitized) > max_len:
    #     sanitized = sanitized[:max_len]
    return sanitized

def get_pdf_path(url, base_output_dir):
    """Generates a valid file path for the PDF based on the URL structure."""
    parsed_url = urlparse(url)
    path = parsed_url.path

    # Remove the base path prefix if it exists
    if path.startswith(BASE_PATH):
        path = path[len(BASE_PATH):]

    # Remove leading/trailing slashes
    path = path.strip('/')

    # If path is empty, it's the root page
    if not path:
        filename = "_root_index.pdf"
        dir_path = base_output_dir
    else:
        # Split path into components
        parts = path.split('/')

        # Check if the last part looks like a file (contains '.')
        if '.' in parts[-1]:
            filename_base = parts[-1].rsplit('.', 1)[0] # Get name without extension
            dir_parts = parts[:-1]
        else:
            # Assume it's an index page for a directory
            filename_base = "index"
            dir_parts = parts

        # Sanitize directory parts and filename
        sanitized_dir_parts = [sanitize_filename(part) for part in dir_parts if part]
        sanitized_filename_base = sanitize_filename(filename_base)

        filename = f"{sanitized_filename_base}.pdf"
        dir_path = os.path.join(base_output_dir, *sanitized_dir_parts)

    # Ensure filename isn't empty
    if not filename.replace(".pdf", ""):
         filename = "_malformed_url.pdf" # Handle edge case

    return os.path.join(dir_path, filename)


def save_as_pdf(url, html_content, output_path):
    """Saves the given HTML content as a PDF file."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Configure pdfkit
        config = None
        if WKHTMLTOPDF_PATH:
            config = pdfkit.configuration(wkhtmltopdf=WKHTMLTOPDF_PATH)

        logging.info(f"Converting to PDF: {url} -> {output_path}")
        success = pdfkit.from_string(
            html_content,
            output_path,
            options=PDFKIT_OPTIONS,
            configuration=config,
            verbose=False # Let logging handle verbosity
        )
        if success:
            logging.info(f"Successfully saved: {output_path}")
            return True
        else:
            # pdfkit often returns True even on some errors with quiet mode,
            # but sometimes returns False. Check file existence as fallback.
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                 logging.info(f"Successfully saved (checked file): {output_path}")
                 return True
            else:
                 logging.warning(f"PDF conversion might have failed (returned False or empty file) for: {url}")
                 # Optionally delete empty file
                 if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                     os.remove(output_path)
                 return False

    except OSError as e:
        # Catch wkhtmltopdf execution errors specifically if possible
        if "No wkhtmltopdf executable found" in str(e):
            logging.error("wkhtmltopdf executable not found. Please install it "
                          "and add it to PATH, or set WKHTMLTOPDF_PATH correctly.")
            # Optionally raise a custom exception or exit
            raise RuntimeError("wkhtmltopdf not found") from e
        logging.error(f"OS Error during PDF conversion for {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during PDF conversion for {url}: {e}")
        # Log the specific html content that failed if it's small enough or relevant
        # logging.debug(f"Failing HTML content preview:\n{html_content[:500]}\n...")
        return False


# --- Main Crawling Logic ---

def crawl_and_save(start_url, output_dir):
    """Crawls the website starting from start_url and saves pages as PDFs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    queue = deque([start_url])
    visited_urls = {start_url}
    processed_count = 0
    error_count = 0

    headers = {'User-Agent': USER_AGENT}

    while queue:
        current_url = queue.popleft()
        logging.info(f"Processing URL ({len(visited_urls)} visited): {current_url}")

        # --- Fetch Page ---
        try:
            response = requests.get(current_url, headers=headers, timeout=30) # Increased timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # Check content type - only process HTML
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logging.info(f"Skipping non-HTML content: {current_url} ({content_type})")
                continue

            # Decode HTML content correctly
            html_content = response.content.decode(response.apparent_encoding or 'utf-8', errors='ignore')


        except requests.exceptions.Timeout:
             logging.warning(f"Timeout fetching {current_url}. Skipping.")
             error_count += 1
             continue
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {current_url}: {e}")
            error_count += 1
            # Optional: Add to a retry queue or just skip
            continue
        except Exception as e:
            logging.error(f"Unexpected error fetching {current_url}: {e}")
            error_count += 1
            continue


        # --- Save as PDF ---
        pdf_path = get_pdf_path(current_url, output_dir)
        if os.path.exists(pdf_path):
            logging.info(f"PDF already exists, skipping save: {pdf_path}")
        else:
            if not save_as_pdf(current_url, html_content, pdf_path):
                error_count += 1
                # Optional: decide if you want to stop finding links if PDF fails
                # continue # Uncomment to skip link finding if PDF save fails

        processed_count += 1

        # --- Find and Enqueue Links ---
        soup = BeautifulSoup(html_content, 'html.parser')
        links_found_on_page = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Resolve relative URLs
            absolute_url = urljoin(current_url, href)
            # Remove URL fragment (#section)
            absolute_url, _ = urldefrag(absolute_url)
            # Parse the absolute URL
            parsed_absolute_url = urlparse(absolute_url)

            # --- Filter Links ---
            # 1. Stay within the same domain
            if parsed_absolute_url.netloc != BASE_DOMAIN:
                # logging.debug(f"Ignoring external domain link: {absolute_url}")
                continue

            # 2. Stay within the specified documentation path
            if not parsed_absolute_url.path.startswith(BASE_PATH):
                # logging.debug(f"Ignoring link outside base path {BASE_PATH}: {absolute_url}")
                continue

            # 3. Ignore links to non-http/https protocols (mailto:, tel:, javascript:)
            if parsed_absolute_url.scheme not in ['http', 'https']:
                # logging.debug(f"Ignoring non-HTTP(S) link: {absolute_url}")
                continue

            # 4. Avoid already visited/queued URLs
            if absolute_url in visited_urls:
                continue

            # --- Add Valid Link to Queue ---
            logging.debug(f"Found valid link: {absolute_url}")
            visited_urls.add(absolute_url)
            queue.append(absolute_url)
            links_found_on_page += 1

        logging.info(f"Found {links_found_on_page} new links on {current_url}")

        # --- Polite Delay ---
        time.sleep(REQUEST_DELAY)

    logging.info(f"\n--- Crawl Finished ---")
    logging.info(f"Total pages processed (attempted PDF conversion): {processed_count}")
    logging.info(f"Total unique URLs visited/added to queue: {len(visited_urls)}")
    logging.info(f"Total errors encountered (fetch/PDF): {error_count}")
    logging.info(f"PDFs saved in: {os.path.abspath(output_dir)}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        crawl_and_save(START_URL, OUTPUT_DIR)
    except RuntimeError as e:
        # Catch specific errors like wkhtmltopdf not found
         logging.critical(f"Critical error, stopping execution: {e}")
    except KeyboardInterrupt:
        logging.info("\n--- Crawl Interrupted by User ---")
    except Exception as e:
        logging.exception("An unexpected error occurred during the crawl.")