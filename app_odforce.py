import os
import requests
import pdfkit
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
import logging
import re
import markdownify
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# --- Configuration ---
START_URL = "https://forums.odforce.net/"
BASE_DOMAIN = urlparse(START_URL).netloc # e.g., 'forums.odforce.net'
BASE_PATH = urlparse(START_URL).path     # e.g., '/'
OUTPUT_DIR = "odforce_scrapMD"
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
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_. ')
    return sanitized

def get_md_path(url, base_output_dir):
    """Generates a valid file path for the Markdown file based on the URL structure."""
    parsed_url = urlparse(url)
    path = parsed_url.path
    if path.startswith(BASE_PATH):
        path = path[len(BASE_PATH):]
    path = path.strip('/')
    if not path:
        filename = "_root_index.md"
        dir_path = base_output_dir
    else:
        parts = path.split('/')
        if '.' in parts[-1]:
            filename_base = parts[-1].rsplit('.', 1)[0]
            dir_parts = parts[:-1]
        else:
            filename_base = "index"
            dir_parts = parts
        sanitized_dir_parts = [sanitize_filename(part) for part in dir_parts if part]
        sanitized_filename_base = sanitize_filename(filename_base)
        filename = f"{sanitized_filename_base}.md"
        dir_path = os.path.join(base_output_dir, *sanitized_dir_parts)
    if not filename.replace(".md", ""):
         filename = "_malformed_url.md"
    return os.path.join(dir_path, filename)

def save_as_markdown(url, post_title, post_content, comments, output_path):
    """Saves the post and its comments as a Markdown file."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# {post_title}\n\n")
            f.write(post_content)
            f.write("\n\n---\n\n")
            for idx, comment in enumerate(comments, 1):
                f.write(f"## Comment {idx}\n")
                f.write(comment)
                f.write("\n\n")
        logging.info(f"Saved Markdown: {output_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to save Markdown for {url}: {e}")
        return False

def extract_post_and_comments(soup):
    """Extracts the main post and all comments from a thread page as Markdown."""
    # This is specific to the odforce forum structure (IPS Community Suite)
    # Main post and comments are in <article> tags with class 'ipsComment'
    posts = soup.find_all('article', class_=re.compile(r'ipsComment'))
    post_title = soup.find('h1', class_=re.compile(r'ipsType_pageTitle'))
    if post_title:
        post_title = post_title.get_text(strip=True)
    else:
        post_title = "Untitled Post"
    post_md = []
    comments_md = []
    for idx, post in enumerate(posts):
        content_div = post.find('div', class_=re.compile(r'cPost_content|ipsType_richText'))
        if content_div:
            html_content = str(content_div)
            md_content = markdownify.markdownify(html_content, heading_style="ATX")
        else:
            md_content = "[No content found]"
        if idx == 0:
            post_md.append(md_content)
        else:
            comments_md.append(md_content)
    return post_title, '\n\n'.join(post_md), comments_md

def is_thread_page(soup):
    """Determines if the page is a thread (post) page."""
    # Thread pages have <article class="ipsComment"> and a post title
    return bool(soup.find('article', class_=re.compile(r'ipsComment'))) and bool(soup.find('h1', class_=re.compile(r'ipsType_pageTitle')))

# --- Main Crawling Logic ---
MAX_CONCURRENT_REQUESTS = 10 # Adjust as needed
# The original REQUEST_DELAY was 0.5s. If we have 10 concurrent workers,
# a small delay per worker helps, but the semaphore is the main rate controller.
DELAY_PER_WORKER = 0.1 # Small delay within each worker task after a request.

async def fetch_html(session, url, headers):
    """Fetches HTML content from a URL asynchronously."""
    try:
        async with session.get(url, headers=headers, timeout=30) as response:
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logging.info(f"Skipping non-HTML content: {url} ({content_type})")
                return None, None
            # Use response.read() to get bytes, then decode
            html_bytes = await response.read()
            # Determine encoding, fallback to utf-8
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

async def process_page_content(executor, html_content, current_url, output_dir, visited_urls_for_links, base_domain_for_links, base_path_for_links):
    """
    Processes HTML content: parses, saves if it's a thread page, and extracts new links.
    Uses executor for CPU-bound (BeautifulSoup) and I/O-bound (file saving) tasks.
    """
    loop = asyncio.get_running_loop()
    soup = await loop.run_in_executor(executor, BeautifulSoup, html_content, 'html.parser')

    new_links_to_add = []
    processed_successfully = False
    saved_this_page = False

    is_thread = await loop.run_in_executor(executor, is_thread_page, soup)
    if is_thread:
        post_title, post_md, comments_md = await loop.run_in_executor(executor, extract_post_and_comments, soup)
        md_path = await loop.run_in_executor(executor, get_md_path, current_url, output_dir)

        # Check existence in executor to avoid blocking
        path_exists = await loop.run_in_executor(executor, os.path.exists, md_path)
        if path_exists:
            logging.info(f"Markdown already exists, skipping save: {md_path}")
            saved_this_page = True # Considered "successful" for processing count
        else:
            saved_this_page = await loop.run_in_executor(executor, save_as_markdown, current_url, post_title, post_md, comments_md, md_path)
    
    processed_successfully = saved_this_page or not is_thread # Successful if saved or not a thread page to begin with

    # Link extraction (can stay in async as soup is already parsed)
    links_found_on_page = 0
    for link in soup.find_all('a', href=True):
        href = link['href']
        # urljoin can be slow if called many times, but for now it's fine here.
        # For extreme optimization, consider if it needs to be in executor.
        absolute_url = urljoin(current_url, href)
        absolute_url, _ = urldefrag(absolute_url)
        parsed_absolute_url = urlparse(absolute_url)

        if parsed_absolute_url.netloc != base_domain_for_links:
            continue
        if not parsed_absolute_url.path.startswith(base_path_for_links):
            continue
        if parsed_absolute_url.scheme not in ['http', 'https']:
            continue
        if absolute_url in visited_urls_for_links: # Check against the shared set
            continue
        if '/topic/' not in parsed_absolute_url.path and '/forum/' not in parsed_absolute_url.path:
            continue
        
        new_links_to_add.append(absolute_url)
        links_found_on_page += 1
    
    if links_found_on_page > 0:
        logging.info(f"Found {links_found_on_page} new links on {current_url}")

    return new_links_to_add, processed_successfully


async def worker(name, queue, session, headers, output_dir, visited_urls, semaphore, executor, base_domain, base_path, stats):
    """Worker that fetches URLs from the queue, processes them, and adds new links back."""
    while True:
        try:
            current_url = await queue.get()
        except asyncio.CancelledError:
            logging.info(f"Worker {name} cancelled.")
            return # Exit if cancelled

        if current_url is None: # Sentinel value to stop the worker
            queue.put_nowait(None) # Put sentinel back for other workers
            logging.info(f"Worker {name} received sentinel. Shutting down.")
            break

        async with semaphore: # Acquire semaphore before processing
            logging.info(f"Worker {name} processing URL ({stats['visited_count']} visited): {current_url}")
            
            html_content, fetched_url = await fetch_html(session, current_url, headers)
            
            if html_content:
                new_links, processed_page = await process_page_content(executor, html_content, current_url, output_dir, visited_urls, base_domain, base_path)
                if processed_page:
                    stats['processed_count'] += 1
                else:
                    stats['error_count'] += 1 # If save_as_markdown returned false or other processing issue

                for link in new_links:
                    if link not in visited_urls:
                        visited_urls.add(link)
                        stats['visited_count'] = len(visited_urls)
                        await queue.put(link)
            else:
                stats['error_count'] += 1 # Error during fetch

            # Polite delay
            await asyncio.sleep(DELAY_PER_WORKER) 
        
        queue.task_done() # Signal that this queue item is done

async def crawl_and_save(start_url, output_dir):
    if not os.path.exists(output_dir):
        # This initial os.makedirs can stay synchronous as it's a one-time setup
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    queue = asyncio.Queue()
    visited_urls = {start_url} # Set to store all URLs ever added to the queue or visited
    
    await queue.put(start_url)

    headers = {'User-Agent': USER_AGENT}
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    # For CPU-bound (BeautifulSoup) and I/O-bound (file saving) tasks
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS * 2) # More threads for blocking tasks

    # Shared statistics
    stats = {'processed_count': 0, 'error_count': 0, 'visited_count': 1}
    
    # Determine base domain and path once
    parsed_start_url = urlparse(start_url)
    base_domain = parsed_start_url.netloc
    base_path = parsed_start_url.path

    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for i in range(MAX_CONCURRENT_REQUESTS):
            task = asyncio.create_task(worker(f"Worker-{i+1}", queue, session, headers, output_dir, visited_urls, semaphore, executor, base_domain, base_path, stats))
            tasks.append(task)

        # Wait for the queue to be fully processed
        await queue.join() 
        
        # Signal workers to stop by putting None for each
        for _ in range(MAX_CONCURRENT_REQUESTS):
            await queue.put(None)

        # Wait for all worker tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True) # Capture exceptions from workers if any

    executor.shutdown(wait=True)

    logging.info(f"\n--- Crawl Finished ---")
    logging.info(f"Total pages processed (attempted Markdown conversion): {stats['processed_count']}")
    logging.info(f"Total unique URLs added to queue/visited: {stats['visited_count']}")
    logging.info(f"Total errors encountered (fetch/Markdown/processing): {stats['error_count']}")
    logging.info(f"Markdown files saved in: {os.path.abspath(output_dir)}")


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Python 3.7+
        asyncio.run(crawl_and_save(START_URL, OUTPUT_DIR))
    except KeyboardInterrupt:
        logging.info("\n--- Crawl Interrupted by User ---")
    except Exception as e:
        # This will catch exceptions from crawl_and_save if not handled internally
        logging.exception("An unexpected error occurred during the crawl.")