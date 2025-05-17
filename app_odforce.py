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
def crawl_and_save(start_url, output_dir):
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
        try:
            response = requests.get(current_url, headers=headers, timeout=30)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logging.info(f"Skipping non-HTML content: {current_url} ({content_type})")
                continue
            html_content = response.content.decode(response.apparent_encoding or 'utf-8', errors='ignore')
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout fetching {current_url}. Skipping.")
            error_count += 1
            continue
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to fetch {current_url}: {e}")
            error_count += 1
            continue
        except Exception as e:
            logging.error(f"Unexpected error fetching {current_url}: {e}")
            error_count += 1
            continue
        soup = BeautifulSoup(html_content, 'html.parser')
        # --- Save as Markdown if thread page ---
        if is_thread_page(soup):
            post_title, post_md, comments_md = extract_post_and_comments(soup)
            md_path = get_md_path(current_url, output_dir)
            if os.path.exists(md_path):
                logging.info(f"Markdown already exists, skipping save: {md_path}")
            else:
                if not save_as_markdown(current_url, post_title, post_md, comments_md, md_path):
                    error_count += 1
        processed_count += 1
        # --- Find and Enqueue Links ---
        links_found_on_page = 0
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)
            absolute_url, _ = urldefrag(absolute_url)
            parsed_absolute_url = urlparse(absolute_url)
            if parsed_absolute_url.netloc != BASE_DOMAIN:
                continue
            if not parsed_absolute_url.path.startswith(BASE_PATH):
                continue
            if parsed_absolute_url.scheme not in ['http', 'https']:
                continue
            if absolute_url in visited_urls:
                continue
            # Only enqueue thread and forum pages (not user profiles, etc)
            # Thread URLs usually contain '/topic/'
            if '/topic/' not in parsed_absolute_url.path and '/forum/' not in parsed_absolute_url.path:
                continue
            visited_urls.add(absolute_url)
            queue.append(absolute_url)
            links_found_on_page += 1
        logging.info(f"Found {links_found_on_page} new links on {current_url}")
        time.sleep(REQUEST_DELAY)
    logging.info(f"\n--- Crawl Finished ---")
    logging.info(f"Total pages processed (attempted Markdown conversion): {processed_count}")
    logging.info(f"Total unique URLs visited/added to queue: {len(visited_urls)}")
    logging.info(f"Total errors encountered (fetch/Markdown): {error_count}")
    logging.info(f"Markdown files saved in: {os.path.abspath(output_dir)}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        crawl_and_save(START_URL, OUTPUT_DIR)
    except KeyboardInterrupt:
        logging.info("\n--- Crawl Interrupted by User ---")
    except Exception as e:
        logging.exception("An unexpected error occurred during the crawl.")