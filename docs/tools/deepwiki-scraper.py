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
import json
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
    """Fetch a page with retries and browser-like headers"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    for attempt in range(3):
        try:
            response = session.get(url, headers=headers, timeout=30)
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
    """Convert HTML to markdown using html2text - diagrams will be added later"""
    # Use html2text for conversion
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0  # No line wrapping
    markdown = h.handle(html_content)
    
    # NOTE: Mermaid diagram processing is DISABLED
    # Diagrams will be matched and inserted in a separate pass
    # This is because the HTML contains diagrams from ALL pages mixed together
    
    return markdown.strip()
    
    # Original markitdown code (temporarily disabled)
    # try:
    #     # Try markitdown first - create a temporary file since it expects file-like input
    #     import tempfile
    #     with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
    #         f.write(html_content)
    #         temp_path = f.name
    #     
    #     try:
    #         md = MarkItDown()
    #         result = md.convert(temp_path)
    #         markdown = result.text_content
    #     finally:
    #         # Clean up temp file
    #         os.unlink(temp_path)
    #     
    #     return markdown.strip()
    # except Exception as e:
    #     print(f"  Warning: markitdown failed ({e}), falling back to html2text")
    #     # Fallback to html2text
    #     h = html2text.HTML2Text()
    #     h.ignore_links = False
    #     h.body_width = 0  # No line wrapping
    #     markdown = h.handle(html_content)
    #     return markdown.strip()

def extract_mermaid_from_nextjs_data(html_text):
    """Extract mermaid diagram code from Next.js streaming response"""
    mermaid_blocks = []
    
    try:
        # Strategy 1: Look for ```mermaid blocks with escaped newlines (\\n)
        # The HTML contains literal \n escape sequences, not actual newlines
        pattern = r'```mermaid\\n(.*?)```'
        matches = re.finditer(pattern, html_text, re.DOTALL)
        
        for match in matches:
            block = match.group(1)
            
            # Unescape newlines and other escapes
            block = block.replace('\\n', '\n')
            block = block.replace('\\t', '\t')
            block = block.replace('\\"', '"')
            block = block.replace('\\\\', '\\')
            block = block.replace('\\u003c', '<')
            block = block.replace('\\u003e', '>')
            block = block.replace('\\u0026', '&')
            
            block = block.strip()
            if len(block) > 10:
                mermaid_blocks.append(block)
                lines = block.split('\n')
                print(f"  Found mermaid diagram: {lines[0][:50]}... ({len(lines)} lines)")
        
        # Strategy 2: JavaScript string extraction (fallback)
        if not mermaid_blocks:
            print(f"  No fenced mermaid blocks found, trying JavaScript extraction...")
            
            mermaid_starts = ['graph TD', 'graph TB', 'graph LR', 'graph RL', 'graph BT',
                            'flowchart TD', 'flowchart TB', 'flowchart LR',
                            'sequenceDiagram', 'classDiagram']
            
            for start_keyword in mermaid_starts:
                pos = 0
                while True:
                    pos = html_text.find(start_keyword, pos)
                    if pos == -1:
                        break
                    
                    # Look backwards for opening quote
                    search_start = max(0, pos - 20)
                    prefix = html_text[search_start:pos]
                    quote_pos = prefix.rfind('"')
                    
                    if quote_pos == -1:
                        pos += 1
                        continue
                    
                    string_start = search_start + quote_pos + 1
                    
                    # Scan forward for closing quote
                    i = pos
                    while i < len(html_text) and i < pos + 10000:
                        if i > 0 and html_text[i-1] == '\\':
                            i += 1
                            continue
                        
                        if html_text[i] == '"':
                            string_end = i
                            break
                        i += 1
                    else:
                        pos += 1
                        continue
                    
                    # Extract and unescape
                    block = html_text[string_start:string_end]
                    block = block.replace('\\n', '\n')
                    block = block.replace('\\t', '\t')
                    block = block.replace('\\"', '"')
                    block = block.replace('\\\\', '\\')
                    block = block.replace('\\u003c', '<')
                    block = block.replace('\\u003e', '>')
                    block = block.replace('\\u0026', '&')
                    
                    lines = [l for l in block.split('\n') if l.strip()]
                    if len(lines) >= 3:
                        mermaid_blocks.append(block.strip())
                        print(f"  Found JS mermaid diagram: {lines[0][:50]}... ({len(lines)} lines)")
                    
                    pos += 1
        
        # Deduplicate
        unique_blocks = []
        seen = set()
        for block in mermaid_blocks:
            fingerprint = block[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_blocks.append(block)
        
        if unique_blocks:
            print(f"  Extracted {len(unique_blocks)} unique mermaid diagram(s)")
        else:
            print(f"  Warning: No valid mermaid diagrams extracted")
        
        return unique_blocks
        
        if unique_blocks:
            print(f"  Extracted {len(unique_blocks)} unique mermaid diagram(s)")
        else:
            print(f"  Warning: No valid mermaid diagrams extracted")
        
        return unique_blocks
        
    except Exception as e:
        print(f"  Warning: Failed to extract mermaid from page data: {e}")
        import traceback
        traceback.print_exc()
        return []

def inject_mermaid_into_html(soup, mermaid_blocks):
    """Inject mermaid blocks into the HTML before markdown conversion"""
    if not mermaid_blocks:
        return soup
    
    # Since we extract ALL diagrams from the Next.js payload (which includes all pages),
    # but we only want the diagrams for THIS specific page, we have a mismatch.
    # Best approach: Place diagrams after EVERY paragraph, not just headings,
    # to create enough insertion points.
    
    content = soup.find('article') or soup.find('main') or soup.find('body')
    if not content:
        return soup
    
    # Find all insertion points: headings and paragraphs
    insertion_points = []
    
    # Add headings
    for heading in content.find_all(['h2', 'h3', 'h4', 'h5', 'h6']):  # Skip h1 (page title)
        insertion_points.append(heading)
    
    # Add some paragraphs as well (every 2nd paragraph)
    paragraphs = content.find_all('p')
    for i, p in enumerate(paragraphs):
        if i % 2 == 0 and len(p.get_text().strip()) > 50:  # Only substantial paragraphs
            insertion_points.append(p)
    
    # If still not enough points, just bail and place what we can
    if len(insertion_points) == 0:
        print(f"  Warning: No insertion points found for {len(mermaid_blocks)} diagrams")
        return soup
    
    # Limit to reasonable number of diagrams (probably only ~10-20 are actually relevant to this page)
    diagrams_to_place = min(len(mermaid_blocks), len(insertion_points), 50)
    
    # Inject placeholders
    for i in range(diagrams_to_place):
        marker_id = f"MERMAID_PLACEHOLDER_{i}"
        marker_html = f'<p>[[{marker_id}]]</p>'
        marker_tag = BeautifulSoup(marker_html, 'html.parser')
        
        # Insert after this insertion point
        insertion_points[i].insert_after(marker_tag)
        if i < 10:  # Only print first 10 to avoid spam
            print(f"  Placed diagram {i+1} after: {insertion_points[i].get_text()[:40]}")
    
    if diagrams_to_place < len(mermaid_blocks):
        print(f"  Note: Placed {diagrams_to_place}/{len(mermaid_blocks)} diagrams (limiting to page-relevant content)")
    
    return soup

def inject_mermaid_into_markdown(markdown, mermaid_blocks):
    """
    Inject mermaid blocks into markdown using intelligent heuristics.
    Place diagrams after headings that suggest a diagram should follow.
    """
    if not mermaid_blocks:
        return markdown
    
    # Keywords that suggest a diagram should follow this heading
    diagram_keywords = [
        'diagram', 'architecture', 'flow', 'structure', 'pipeline', 
        'workflow', 'overview', 'layers', 'graph', 'visualization',
        'sequence', 'process', 'hierarchy', 'dependency', 'lifecycle'
    ]
    
    lines = markdown.split('\n')
    result = []
    diagram_idx = 0
    
    i = 0
    while i < len(lines) and diagram_idx < len(mermaid_blocks):
        line = lines[i]
        result.append(line)
        
        # Check if this is a heading that should have a diagram
        stripped = line.strip().lower()
        if stripped.startswith('###') or stripped.startswith('##'):
            # Extract heading text
            heading_text = stripped.lstrip('#').strip()
            
            # Check if heading contains diagram keywords
            if any(keyword in heading_text for keyword in diagram_keywords):
                # Look ahead to see if next non-empty line is already a diagram
                next_line_idx = i + 1
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    next_line_idx += 1
                
                # Only inject if next content is not already a diagram
                if next_line_idx < len(lines) and not lines[next_line_idx].strip().startswith('```'):
                    # Insert diagram after this heading
                    result.append('')
                    result.append('```mermaid')
                    result.append(mermaid_blocks[diagram_idx])
                    result.append('```')
                    result.append('')
                    diagram_idx += 1
                    print(f"  [DEBUG] Inserted diagram {diagram_idx} after heading: {heading_text[:50]}")
        
        i += 1
    
    # Append remaining lines
    while i < len(lines):
        result.append(lines[i])
        i += 1
    
    # If we still have unused diagrams, append them at the end
    if diagram_idx < len(mermaid_blocks):
        print(f"  [DEBUG] {len(mermaid_blocks) - diagram_idx} diagrams not placed, appending at end")
        result.append('')
        result.append('## Additional Diagrams')
        result.append('')
        for idx in range(diagram_idx, len(mermaid_blocks)):
            result.append('```mermaid')
            result.append(mermaid_blocks[idx])
            result.append('```')
            result.append('')
    
    return '\n'.join(result)

def extract_page_content(url, session):
    """Extract the main content from a wiki page"""
    print(f"  Fetching {url}...")
    response = fetch_page(url, session)
    
    # NOTE: Mermaid diagrams are client-side rendered and cannot be extracted
    # without running JavaScript. They exist in the Next.js data payload but
    # are mixed with all other pages, making per-page extraction impossible.
    
    # Parse HTML with BeautifulSoup to get THIS page's structure
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
    
    # DEBUG: Check if mermaid blocks exist in HTML
    mermaid_count = html_content.count('language-mermaid')
    print(f"  [DEBUG] Found {mermaid_count} mermaid blocks in HTML content")
    
    markdown = convert_html_to_markdown(html_content)
    
    # NOTE: Mermaid diagram injection disabled - diagrams are mixed across all pages
    # in the JavaScript payload and cannot be reliably extracted per-page
    
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

def extract_and_enhance_diagrams(repo, output_dir, session):
    """Extract diagrams from JavaScript and enhance all markdown files."""
    print("\n" + "="*80)
    print("PHASE 2: Extracting diagrams and enhancing markdown files")
    print("="*80)
    
    # Extract all diagrams first (fetch from any page - they're all in the JS)
    print("\nExtracting diagrams from JavaScript payload...")
    url = f'https://deepwiki.com/{repo}/1-overview'
    
    try:
        response = session.get(url)
        response.raise_for_status()
        html_text = response.text
    except Exception as e:
        print(f"  Warning: Could not fetch diagrams: {e}")
        return
    
    # Extract diagrams with context
    diagram_pattern = r'```mermaid\\n(.*?)```'
    all_diagrams = re.findall(diagram_pattern, html_text, re.DOTALL)
    print(f"  Found {len(all_diagrams)} total diagrams")
    
    # Extract with more context (500+ chars before each diagram)
    diagram_contexts = []
    markdown_pattern = r'([^`]{500,}?)```mermaid\\n(.*?)```'
    markdown_matches = re.finditer(markdown_pattern, html_text, re.DOTALL)
    
    for match in markdown_matches:
        context_before = match.group(1)
        diagram = match.group(2)
        
        # Unescape context - keep last 500 chars
        context = context_before.replace('\\n', '\n')
        context = context.replace('\\t', ' ')
        context = context.replace('\\"', '"')
        context = context.replace('\\\\', '\\')
        context = context.replace('\\u003c', '<')
        context = context.replace('\\u003e', '>')
        context = context.replace('\\u0026', '&')
        context = context[-500:].strip()
        
        # Unescape diagram
        diagram = diagram.replace('\\n', '\n')
        diagram = diagram.replace('\\t', '\t')
        diagram = diagram.replace('\\"', '"')
        diagram = diagram.replace('\\\\', '\\')
        diagram = diagram.replace('\\u003c', '<')
        diagram = diagram.replace('\\u003e', '>')
        diagram = diagram.replace('\\u0026', '&')
        diagram = diagram.strip()
        
        if len(diagram) > 10:
            context_lines = [l.strip() for l in context.split('\n') if l.strip()]
            
            # Find last heading
            last_heading = None
            for line in reversed(context_lines):
                if line.startswith('#'):
                    last_heading = line
                    break
            
            # Get last 2-3 non-heading lines as anchor text
            anchor_lines = []
            for line in reversed(context_lines):
                if not line.startswith('#') and len(line) > 20:
                    anchor_lines.insert(0, line)
                    if len(anchor_lines) >= 3:
                        break
            
            anchor_text = ' '.join(anchor_lines)[-300:] if anchor_lines else ''
            
            diagram_contexts.append({
                'last_heading': last_heading or '',
                'anchor_text': anchor_text,
                'diagram': diagram
            })
    
    print(f"  Found {len(diagram_contexts)} diagrams with context")
    
    # Save diagrams for reference
    diagram_file = output_dir / '_diagrams_with_context.txt'
    with open(diagram_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(diagram_contexts, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"DIAGRAM {i}\n")
            f.write(f"Heading: {item['last_heading']}\n")
            f.write(f"Anchor: {item['anchor_text'][:200]}...\n")
            f.write(f"{'='*80}\n")
            f.write(f"```mermaid\n{item['diagram']}\n```\n")
    
    print(f"  Saved diagram reference to {diagram_file.name}")
    
    # Now enhance all markdown files
    print("\nEnhancing markdown files with diagrams...")
    md_files = list(output_dir.glob('**/*.md'))
    md_files = [f for f in md_files if not f.name.startswith('_')]
    
    enhanced_count = 0
    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Skip if already has diagrams
        if '```mermaid' in content:
            continue
        
        # Match and insert diagrams
        lines = content.split('\n')
        diagrams_used = set()
        pending_insertions = []
        
        # Normalize content for matching
        content_normalized = content.lower()
        content_normalized = ' '.join(content_normalized.split())
        
        for idx, item in enumerate(diagram_contexts):
            if idx in diagrams_used:
                continue
            
            anchor = item['anchor_text']
            heading = item['last_heading']
            
            if not anchor and not heading:
                continue
            
            best_match_line = -1
            best_match_score = 0
            
            # Try anchor text matching
            if len(anchor) > 50:
                anchor_normalized = anchor.lower()
                anchor_normalized = ' '.join(anchor_normalized.split())
                
                for chunk_size in [300, 200, 150, 100, 80]:
                    if len(anchor_normalized) >= chunk_size:
                        test_chunk = anchor_normalized[-chunk_size:]
                        pos = content_normalized.find(test_chunk)
                        if pos != -1:
                            # Convert char position to line number
                            char_count = 0
                            for line_num, line in enumerate(lines):
                                char_count += len(' '.join(line.split())) + 1
                                if char_count >= pos:
                                    best_match_line = line_num
                                    best_match_score = chunk_size
                                    break
                            if best_match_line != -1:
                                break
            
            # Fallback: heading match
            if best_match_line == -1 and heading:
                heading_normalized = heading.lower().replace('#', '').strip()
                heading_normalized = ' '.join(heading_normalized.split())
                
                for line_num, line in enumerate(lines):
                    if line.strip().startswith('#'):
                        line_normalized = line.lower().replace('#', '').strip()
                        line_normalized = ' '.join(line_normalized.split())
                        
                        if heading_normalized in line_normalized:
                            best_match_line = line_num
                            best_match_score = 50
                            break
            
            if best_match_line != -1:
                # Find insertion point: after paragraph
                insert_line = best_match_line + 1
                
                if lines[best_match_line].strip().startswith('#'):
                    # Skip blank lines after heading
                    while insert_line < len(lines) and not lines[insert_line].strip():
                        insert_line += 1
                    # Skip through paragraph
                    while insert_line < len(lines):
                        if not lines[insert_line].strip() or lines[insert_line].strip().startswith('#'):
                            break
                        insert_line += 1
                else:
                    # Find end of current paragraph
                    while insert_line < len(lines):
                        if not lines[insert_line].strip() or lines[insert_line].strip().startswith('#'):
                            break
                        insert_line += 1
                
                pending_insertions.append((insert_line, item['diagram'], best_match_score, idx))
                diagrams_used.add(idx)
        
        # Insert diagrams (from bottom up)
        if pending_insertions:
            pending_insertions.sort(key=lambda x: x[0], reverse=True)
            
            for insert_line, diagram, score, idx in pending_insertions:
                lines.insert(insert_line, '')
                lines.insert(insert_line, '```')
                lines.insert(insert_line, diagram)
                lines.insert(insert_line, '```mermaid')
                lines.insert(insert_line, '')
            
            # Save enhanced file
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            enhanced_count += 1
            print(f"  ✓ {md_file.name} ({len(pending_insertions)} diagrams)")
    
    print(f"\n✓ Enhanced {enhanced_count} files with diagrams")

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
    
    print("="*80)
    print("PHASE 1: Extracting clean markdown content")
    print("="*80)
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
        
        # Extract and enhance with diagrams
        extract_and_enhance_diagrams(repo, output_dir, session)
        
        print("\n" + "="*80)
        print("✓ COMPLETE: All pages extracted and enhanced with diagrams")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
