#!/usr/bin/env python3
"""
DeepWiki Scraper - Extract wiki pages from deepwiki.com as markdown files

Usage:
    python deepwiki-scraper.py <owner/repo> <output-dir>

Example:
    python deepwiki-scraper.py jzombie/rust-llkv ./wiki-output
"""

import sys
import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown
import html2text

def sanitize_filename(text):
    """Convert text to a safe filename"""
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-').lower()

def fetch_page(url, session):
    """Fetch a page with retries"""
    for attempt in range(3):
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            if attempt == 2:
                raise
            print(f"  Retry {attempt + 1}/3 after error: {e}")
            time.sleep(2)

def discover_subsections(repo, main_page_num, session):
    """Try to discover subsections for a main page by checking common patterns"""
    base_url = f"https://deepwiki.com/{repo}"
    subsections = []
    
    # Try up to 10 subsections (e.g., 2.1, 2.2, ..., 2.10)
    for sub_num in range(1, 11):
        test_url = f"{base_url}/{main_page_num}-{sub_num}-"
        try:
            response = session.head(test_url, allow_redirects=True, timeout=5)
            if response.status_code == 200:
                # Follow redirect to get actual URL
                actual_url = response.url
                # Extract title from URL slug
                match = re.search(r'/(\d+-\d+)-(.+)$', urlparse(actual_url).path)
                if match:
                    page_num = match.group(1).replace('-', '.')
                    title_slug = match.group(2)
                    # Convert slug to title (best guess)
                    title = title_slug.replace('-', ' ').title()
                    
                    subsections.append({
                        'number': page_num,
                        'title': title,
                        'url': actual_url,
                        'href': urlparse(actual_url).path,
                        'level': 1
                    })
        except:
            # Connection error or timeout, stop trying
            break
    
    return subsections

def extract_wiki_structure(repo, session):
    """Extract the complete wiki structure including subsections from the main wiki page"""
    base_url = f"https://deepwiki.com/{repo}"
    
    print(f"Fetching wiki structure from {base_url}...")
    response = fetch_page(base_url, session)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    pages = []
    
    # Find all links that match the page pattern (including subsections with dots)
    # Pattern: /owner/repo/1-overview, /owner/repo/2.1-subsection, etc.
    all_links = soup.find_all('a', href=re.compile(f'^/{repo}/\d+'))
    
    seen_urls = set()
    for link in all_links:
        href = link.get('href', '')
        if not href or href in seen_urls:
            continue
            
        # Extract page number and title
        # Expected format: /owner/repo/1-overview or /owner/repo/2.1-subsection
        match = re.search(r'/(\d+(?:\.\d+)*)-(.+)$', href)
        if match:
            page_num = match.group(1)
            title = link.get_text(strip=True)
            full_url = urljoin(base_url, href)
            
            # Determine if this is a subsection based on page number
            level = page_num.count('.')
            
            pages.append({
                'number': page_num,
                'title': title,
                'url': full_url,
                'href': href,
                'level': level  # 0 for main pages (1, 2, 3), 1 for subsections (2.1, 2.2)
            })
            seen_urls.add(href)
    
    # Sort by page number (properly handling subsections)
    def sort_key(page):
        parts = [int(x) for x in page['number'].split('.')]
        return parts
    
    pages.sort(key=sort_key)
    
    return pages

def convert_html_to_markdown(html_content):
    """Convert HTML content to markdown, trying markitdown first, then html2text"""
    # Try markitdown first
    try:
        md = MarkItDown()
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
            f.write(html_content)
            temp_path = f.name
        
        try:
            result = md.convert(temp_path)
            return result.text_content.strip()
        finally:
            os.unlink(temp_path)
    except Exception:
        # Fallback to html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = False
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        h.skip_internal_links = False
        
        markdown = h.handle(html_content)
        return markdown.strip()

def extract_page_content(url, session):
    """Extract the main content from a wiki page"""
    print(f"  Fetching {url}...")
    response = fetch_page(url, session)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove unwanted elements first
    for elem in soup.select('nav, header, footer, aside, .sidebar, .menu, script, style, .navigation, [role="navigation"]'):
        elem.decompose()
    
    # Find main content area
    content = None
    
    # Try common content selectors
    for selector in ['article', 'main', '.wiki-content', '.content', '#content', '.markdown-body']:
        content = soup.select_one(selector)
        if content:
            break
    
    # Try role attribute separately
    if not content:
        content = soup.find(attrs={'role': 'main'})
    
    # Fallback: look for the largest text block
    if not content:
        content = soup.find('body')
    
    if not content:
        raise Exception(f"Could not find content on page: {url}")
    
    # Remove DeepWiki header/UI elements before main content
    # Look for specific text patterns and remove those elements
    for elem in content.find_all(['div', 'span', 'a', 'button']):
        text = elem.get_text(strip=True)
        # Remove elements containing these DeepWiki UI strings
        if any(keyword in text for keyword in [
            'Index your code with Devin',
            'Edit Wiki',
            'Last indexed:',
            'View this search on DeepWiki'
        ]) and len(text) < 200:  # Only remove short elements (not whole paragraphs)
            elem.decompose()
    
    # Remove specific DeepWiki navigation elements (table of contents list)
    # This is typically a long list of links to all wiki pages
    for ul in content.find_all('ul'):
        # Check if this looks like a navigation menu (many links to wiki pages)
        links = ul.find_all('a')
        if len(links) > 5:  # If more than 5 links in one list
            # Check if they're internal wiki links
            wiki_links = [a for a in links if a.get('href', '').startswith('/')]
            if len(wiki_links) > len(links) * 0.8:  # If 80%+ are internal links
                ul.decompose()
    
    # Convert to markdown
    html_content = str(content)
    markdown = convert_html_to_markdown(html_content)
    
    # Clean up markdown: remove duplicate titles and stray "Menu" lines
    lines = markdown.split('\n')
    clean_lines = []
    seen_title = False
    
    for line in lines:
        stripped = line.strip()
        
        # Skip standalone "Menu"
        if stripped == 'Menu':
            continue
        
        # Skip duplicate titles (keep first occurrence)
        if line.startswith('# '):
            if seen_title and line == clean_lines[0]:
                continue  # Skip duplicate
            seen_title = True
        
        clean_lines.append(line)
    
    markdown = '\n'.join(clean_lines).strip()
    
    # Fix internal wiki links to match our filename structure
    # Convert /jzombie/rust-llkv/2.1-page to section-2/2-1-page.md
    def fix_wiki_link(match):
        full_path = match.group(1)
        # Extract page number and slug (full_path is just "4-query-planning" part)
        link_match = re.search(r'^(\d+(?:\.\d+)*)-(.+)$', full_path)
        if link_match:
            page_num = link_match.group(1)
            slug = link_match.group(2)
            
            # Convert page number format: 2.1 -> 2-1
            file_num = page_num.replace('.', '-')
            
            # Determine if it's a subsection
            if '.' in page_num:
                main_section = page_num.split('.')[0]
                return f'](section-{main_section}/{file_num}-{slug}.md)'
            else:
                return f']({file_num}-{slug}.md)'
        return match.group(0)
    
    # Replace wiki links: [text](/owner/repo/page) -> [text](file.md)
    markdown = re.sub(r'\]\(/[^/]+/[^/]+/([^)]+)\)', fix_wiki_link, markdown)
    
    return markdown

def main():
    if len(sys.argv) < 3:
        print("Usage: python deepwiki-scraper.py <owner/repo> <output-dir>")
        print("Example: python deepwiki-scraper.py jzombie/rust-llkv ./wiki-output")
        sys.exit(1)
    
    repo = sys.argv[1]
    output_dir = Path(sys.argv[2])
    
    # Validate repo format
    if not re.match(r'^[\w-]+/[\w-]+$', repo):
        print("Error: Repository must be in format 'owner/repo'")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create session with headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    try:
        # Extract wiki structure
        pages = extract_wiki_structure(repo, session)
        
        if not pages:
            print("\nError: No wiki pages found")
            print("The repository may not have a DeepWiki wiki, or the HTML structure has changed.")
            sys.exit(1)
        
        print(f"\nFound {len(pages)} pages\n")
        
        # Create directory structure for subsections
        # Main pages go in root, subsections in subdirectories
        main_pages = [p for p in pages if p['level'] == 0]
        print(f"Main pages: {len(main_pages)}")
        print(f"Subsections: {len(pages) - len(main_pages)}\n")
        
        # Extract each page
        success_count = 0
        for page in pages:
            try:
                markdown = extract_page_content(page['url'], session)
                
                # Generate filename based on hierarchy
                num_prefix = page['number'].replace('.', '-')
                title_slug = sanitize_filename(page['title'])
                
                # Determine output path based on level
                if page['level'] == 0:
                    # Main page: goes in root directory
                    filename = f"{num_prefix}-{title_slug}.md"
                    filepath = output_dir / filename
                else:
                    # Subsection: create subdirectory named after main section
                    main_section = page['number'].split('.')[0]
                    section_dir = output_dir / f"section-{main_section}"
                    section_dir.mkdir(exist_ok=True)
                    filename = f"{num_prefix}-{title_slug}.md"
                    filepath = section_dir / filename
                
                # Ensure content starts with title
                if not markdown.startswith('#'):
                    markdown = f"# {page['title']}\n\n{markdown}"
                
                # Write file
                filepath.write_text(markdown, encoding='utf-8')
                print(f"  ✓ {filepath.relative_to(output_dir)} ({len(markdown)} bytes)")
                success_count += 1
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                print(f"  ✗ Failed to extract {page['title']}: {e}")
        
        print(f"\n✓ Successfully extracted {success_count}/{len(pages)} pages to {output_dir}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
