import os
# import requests # requests is not used when using aiohttp directly
# import pdfkit # Will be removed
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
# from collections import deque # Will be replaced by asyncio.Queue (already done)
import logging
import re
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import markdownify # Added for Markdown conversion

# --- Configuration ---
START_URL = "https://www.sidefx.com/docs/houdini/"
BASE_DOMAIN = urlparse(START_URL).netloc # e.g., 'www.sidefx.com'
BASE_PATH = urlparse(START_URL).path     # e.g., '/docs/houdini/'
OUTPUT_DIR = "houdini_docs_mkdown" # Changed output directory name
# REQUEST_DELAY is handled by DELAY_PER_WORKER and semaphore
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# PDFKIT_OPTIONS and WKHTMLTOPDF_PATH are removed as they are PDF-specific

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

def get_md_path(url, base_output_dir):
    """Generates a valid file path for the Markdown file based on the URL structure."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path.startswith(BASE_PATH):
        path = path[len(BASE_PATH):]
    path = path.strip('/')
    if not path: # Root page
        filename = "_root_index.md"
        dir_path = base_output_dir
    else:
        parts = path.split('/')
        # Determine if the last part is a filename (e.g., ends with .html) or a directory
        if parts[-1] and ( '.' in parts[-1] and not parts[-1].endswith('/') ) : # Likely a file
            filename_base = parts[-1].rsplit('.', 1)[0]
            dir_parts = parts[:-1]
        else: # Likely a directory, create an index.md
            filename_base = "index"
            # If path ends with /, last part is empty after strip, or it's a dir name
            dir_parts = [p for p in parts if p] # Filter out empty parts from trailing slashes
        
        sanitized_dir_parts = [sanitize_filename(part) for part in dir_parts if part]
        sanitized_filename_base = sanitize_filename(filename_base)
        filename = f"{sanitized_filename_base}.md"
        dir_path = os.path.join(base_output_dir, *sanitized_dir_parts)
    
    # Ensure filename isn't just ".md"
    if not filename.replace(".md", "").strip():
         filename = "_malformed_url.md"
    return os.path.join(dir_path, filename)

# --- Main Crawling Logic ---
MAX_CONCURRENT_REQUESTS = 10 # Markdown conversion is lighter than PDF
DELAY_PER_WORKER = 0.1

async def fetch_html(session, url, headers):
    """Fetches HTML content from a URL asynchronously."""
    try:
        async with session.get(url, headers=headers, timeout=30) as response:
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logging.info(f"Skipping non-HTML content: {url} ({content_type})")
                return None, url
            html_bytes = await response.read()
            encoding = response.charset or 'utf-8'
            html_content = html_bytes.decode(encoding, errors='ignore')
            return html_content, url
    except asyncio.TimeoutError:
        logging.warning(f"Timeout fetching {url}. Skipping.")
        return None, url
    except aiohttp.ClientError as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None, url
    except Exception as e:
        logging.error(f"Unexpected error fetching {url}: {e}")
        return None, url

def save_as_markdown_executor(url, html_content, output_path):
    """Converts HTML to Markdown and saves it. Designed to be run in an executor."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # SideFX specific: Attempt to extract main content only for cleaner Markdown
        temp_soup = BeautifulSoup(html_content, 'html.parser')
        main_content_div = temp_soup.find('div', id='main-content') # Common ID for main content
        if not main_content_div:
            main_content_div = temp_soup.find('main') # HTML5 main tag
        # Add more selectors if needed, e.g., temp_soup.select_one('.content-class')
        
        html_to_convert = str(main_content_div) if main_content_div else html_content
        
        md_content = markdownify.markdownify(html_to_convert, heading_style="ATX", strip=['script', 'style'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"<!-- Original URL: {url} -->\n\n") # Add original URL as a comment
            f.write(md_content)
        logging.info(f"Successfully saved Markdown: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save Markdown for {url} to {output_path}: {e}")
        return False

async def process_page_content_md(executor, html_content, current_url, output_dir, visited_urls_for_links, base_domain_for_links, base_path_for_links):
    """Processes HTML: saves as Markdown (in executor), parses for links (in executor)."""
    loop = asyncio.get_running_loop()
    new_links_to_add = []
    md_saved_or_exists = False

    md_path = get_md_path(current_url, output_dir)

    path_exists = await loop.run_in_executor(executor, os.path.exists, md_path)
    if path_exists:
        logging.info(f"Markdown already exists, skipping save: {md_path}")
        md_saved_or_exists = True
    else:
        try:
            md_saved_or_exists = await loop.run_in_executor(
                executor, 
                save_as_markdown_executor, 
                current_url, 
                html_content, 
                md_path
            )
        except Exception as e:
            logging.error(f"Error calling save_as_markdown_executor for {current_url}: {e}")
            md_saved_or_exists = False

    # --- Find and Enqueue Links from original full HTML--- 
    # Link extraction should use the original full html_content, not just the main part for conversion accuracy
    soup = await loop.run_in_executor(executor, BeautifulSoup, html_content, 'html.parser')
    links_found_on_page = 0
    for link in soup.find_all('a', href=True):
        href = link['href']
        absolute_url = urljoin(current_url, href)
        absolute_url, _ = urldefrag(absolute_url)
        parsed_absolute_url = urlparse(absolute_url)

        if parsed_absolute_url.netloc != base_domain_for_links:
            continue
        if not parsed_absolute_url.path.startswith(base_path_for_links):
            continue
        if parsed_absolute_url.scheme not in ['http', 'https']:
            continue
        if absolute_url == current_url or absolute_url in visited_urls_for_links:
            continue
        
        new_links_to_add.append(absolute_url)
        links_found_on_page += 1
    
    if links_found_on_page > 0:
        logging.debug(f"Found {links_found_on_page} new links on {current_url}")

    return new_links_to_add, md_saved_or_exists # Return based on if MD was saved or existed

async def worker(name, queue, session, headers, output_dir, visited_urls, semaphore, executor, base_domain, base_path, stats):
    """Worker that fetches URLs, processes them for Markdown, and adds new links."""
    while True:
        try:
            current_url = await queue.get()
        except asyncio.CancelledError:
            logging.info(f"Worker {name} cancelled.")
            return

        if current_url is None: # Sentinel
            queue.put_nowait(None) # Ensure other workers see sentinel
            logging.info(f"Worker {name} received sentinel. Shutting down.")
            break

        async with semaphore:
            logging.info(f"W:{name} Q:{queue.qsize()} V:{stats['visited_count']} P:{stats['processed_count']} E:{stats['error_count']} > {current_url}")
            
            html_content, fetched_url = await fetch_html(session, current_url, headers)
            
            if html_content:
                try:
                    new_links, processed_page_md = await process_page_content_md(
                        executor, html_content, current_url, output_dir, 
                        visited_urls, base_domain, base_path
                    )
                    if processed_page_md:
                        stats['processed_count'] += 1
                    else:
                        # This implies saving markdown failed for a new page
                        stats['error_count'] += 1 

                    for link in new_links:
                        if link not in visited_urls: # Check before adding to queue and set
                            visited_urls.add(link)
                            stats['visited_count'] = len(visited_urls)
                            await queue.put(link)
                except Exception as e:
                    logging.error(f"Critical error in worker {name} processing {current_url}: {e}", exc_info=True)
                    stats['error_count'] += 1
            else:
                stats['error_count'] += 1 # Error during fetch

            await asyncio.sleep(DELAY_PER_WORKER) 
        
        queue.task_done()

async def crawl_and_save(start_url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    queue = asyncio.Queue()
    visited_urls = {start_url} 
    await queue.put(start_url)

    headers = {'User-Agent': USER_AGENT}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    # More threads for I/O and CPU bound tasks (parsing, file writing)
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS * 2) 

    stats = {'processed_count': 0, 'error_count': 0, 'visited_count': 1}
    parsed_start_url = urlparse(start_url)
    base_domain = parsed_start_url.netloc
    base_path = parsed_start_url.path
    
    all_tasks_completed_normally = True
    worker_tasks = []

    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            for i in range(MAX_CONCURRENT_REQUESTS):
                # Pass worker ID as string for logging clarity
                task = asyncio.create_task(worker(f"{i+1}", queue, session, headers, output_dir, visited_urls, semaphore, executor, base_domain, base_path, stats))
                worker_tasks.append(task)

            await queue.join() # Wait for all items initially in queue to be processed
            
            logging.info("Main queue processed. Sending stop sentinels to workers.")
            for _ in range(MAX_CONCURRENT_REQUESTS):
                await queue.put(None) 

            results = await asyncio.gather(*worker_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"Worker task {i+1} resulted in an exception: {result}", exc_info=result)
                    all_tasks_completed_normally = False
                elif result is None: # Should not happen if tasks complete normally without error
                    logging.warning(f"Worker task {i+1} returned None unexpectedly.")

    except Exception as e: 
        logging.critical(f"Crawl aborted due to critical error in main management: {e}", exc_info=True)
        all_tasks_completed_normally = False
        for task in worker_tasks:
            if not task.done():
                task.cancel()
        # Await cancellations to allow them to finish cleanly if possible
        await asyncio.gather(*worker_tasks, return_exceptions=True) 
    finally:
        logging.info("Shutting down thread pool executor.")
        executor.shutdown(wait=True)
        logging.info("Thread pool executor shut down.")

    logging.info(f"\n--- Crawl Finished ---")
    if not all_tasks_completed_normally:
        logging.warning("Crawl finished, but one or more errors or unexpected issues occurred.")
    logging.info(f"Total pages processed (attempted Markdown conversion): {stats['processed_count']}")
    logging.info(f"Total unique URLs added to queue/visited: {stats['visited_count']}")
    logging.info(f"Total errors encountered (fetch/Markdown/processing): {stats['error_count']}")
    logging.info(f"Markdown files saved in: {os.path.abspath(output_dir)}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        asyncio.run(crawl_and_save(START_URL, OUTPUT_DIR))
    except KeyboardInterrupt:
        logging.info("\n--- Crawl Interrupted by User --- Processing will stop after current tasks.")
        # Note: asyncio.run handles KeyboardInterrupt and attempts to clean up tasks.
        # For more graceful shutdown, one might use asyncio.Event to signal workers.
    except Exception as e:
        logging.exception("An unexpected error occurred at the top level of the crawl.")