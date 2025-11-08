#!/usr/bin/env python3
"""
Two-pass wiki extraction:
1. Extract all diagrams from JavaScript payload
2. Extract clean content per page
3. Match and insert diagrams by context
"""

import os
import sys
import re
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import html2text

def extract_all_diagrams():
    """Extract ALL mermaid diagrams from the JavaScript payload of any wiki page"""
    url = "https://deepwiki.com/jzombie/rust-llkv/1-overview"
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    })
    
    print("Pass 1: Extracting all diagrams from JavaScript...")
    response = session.get(url, timeout=30)
    html_text = response.text
    
    # Extract mermaid diagrams with context
    # Pattern: ```mermaid\\n...```
    pattern = r'```mermaid\\n(.*?)```'
    matches = re.finditer(pattern, html_text, re.DOTALL)
    
    diagrams = []
    for match in matches:
        block = match.group(1)
        # Unescape
        block = block.replace('\\n', '\n')
        block = block.replace('\\t', '\t')
        block = block.replace('\\"', '"')
        block = block.replace('\\\\', '\\')
        block = block.replace('\\u003c', '<')
        block = block.replace('\\u003e', '>')
        block = block.replace('\\u0026', '&')
        
        block = block.strip()
        if len(block) > 10:
            diagrams.append(block)
    
    print(f"  Found {len(diagrams)} diagrams")
    
    # Extract much more context (500+ chars before each diagram) for accurate matching
    diagram_contexts = []
    
    # Extract markdown chunks that contain diagrams - get LOTS of context
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
        
        # Keep substantial context
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
            # Extract multiple potential anchor points from context
            # - Last heading (##, ###)
            # - Last sentence before diagram
            # - Distinctive phrases (e.g., "The following diagram", "Sources:")
            
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
                'full_context': context,
                'last_heading': last_heading or '',
                'anchor_text': anchor_text,
                'diagram': diagram,
                'first_line': diagram.split('\n')[0]
            })
    
    print(f"  Found {len(diagram_contexts)} diagrams with context")
    return diagram_contexts

def match_and_insert_diagrams(markdown_content, diagram_contexts):
    """Match diagrams to markdown content using fuzzy matching on anchor text"""
    lines = markdown_content.split('\n')
    diagrams_used = set()
    pending_insertions = []  # (line_num, diagram, score)
    
    # Normalize content for matching
    content_normalized = markdown_content.lower()
    content_normalized = ' '.join(content_normalized.split())  # Normalize whitespace
    
    # For each diagram, find best match location
    for idx, item in enumerate(diagram_contexts):
        anchor = item['anchor_text']
        heading = item['last_heading']
        
        if not anchor and not heading:
            continue
        
        best_match_line = -1
        best_match_score = 0
        
        # Try anchor text matching with progressively smaller chunks
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
        
        # Fallback: try heading match
        if best_match_line == -1 and heading:
            heading_normalized = heading.lower().replace('#', '').strip()
            heading_normalized = ' '.join(heading_normalized.split())
            
            for line_num, line in enumerate(lines):
                if line.strip().startswith('#'):
                    line_normalized = line.lower().replace('#', '').strip()
                    line_normalized = ' '.join(line_normalized.split())
                    
                    if heading_normalized in line_normalized:
                        best_match_line = line_num
                        best_match_score = 50  # Lower score for heading-only match
                        break
        
        if best_match_line != -1:
            # Find insertion point: after paragraph following the match
            insert_line = best_match_line + 1
            
            # If matched line is a heading, skip past it and find end of first paragraph
            if lines[best_match_line].strip().startswith('#'):
                # Skip blank lines after heading
                while insert_line < len(lines) and not lines[insert_line].strip():
                    insert_line += 1
                
                # Skip through paragraph (until blank or next heading)
                while insert_line < len(lines):
                    if not lines[insert_line].strip() or lines[insert_line].strip().startswith('#'):
                        break
                    insert_line += 1
            else:
                # Matched in middle of text - find end of current paragraph
                while insert_line < len(lines):
                    if not lines[insert_line].strip() or lines[insert_line].strip().startswith('#'):
                        break
                    insert_line += 1
            
            pending_insertions.append((insert_line, item['diagram'], best_match_score, idx))
            diagrams_used.add(idx)
            print(f"  Matched diagram {idx+1} (score={best_match_score}) at line {best_match_line}")
    
    # Sort by line number (descending) to insert from bottom up
    pending_insertions.sort(key=lambda x: x[0], reverse=True)
    
    # Insert diagrams
    for insert_line, diagram, score, idx in pending_insertions:
        lines.insert(insert_line, '')
        lines.insert(insert_line, '```')
        lines.insert(insert_line, diagram)
        lines.insert(insert_line, '```mermaid')
        lines.insert(insert_line, '')
    
    print(f"  Inserted {len(diagrams_used)} diagrams total")
    return '\n'.join(lines)

def main():
    # Extract all diagrams first
    diagram_contexts = extract_all_diagrams()
    
    # Save for reference
    output_dir = Path('/output')
    diagram_file = output_dir / '_diagrams_with_context.txt'
    
    with open(diagram_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(diagram_contexts, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"DIAGRAM {i}\n")
            f.write(f"Heading: {item['last_heading']}\n")
            f.write(f"Anchor: {item['anchor_text'][:200]}...\n")
            f.write(f"{'='*80}\n")
            f.write(f"```mermaid\n{item['diagram']}\n```\n")
    
    print(f"\nSaved diagrams to {diagram_file}")
    
    # Process ALL markdown files
    print("\nEnhancing all markdown files...")
    md_files = list(output_dir.glob('**/*.md'))
    md_files = [f for f in md_files if not f.name.startswith('_')]  # Skip helper files
    
    enhanced_count = 0
    for md_file in md_files:
        print(f"\nProcessing {md_file.name}...")
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if already has diagrams
        if '```mermaid' in content:
            print(f"  Already has diagrams, skipping")
            continue
        
        enhanced = match_and_insert_diagrams(content, diagram_contexts)
        
        # Only save if we added diagrams
        if '```mermaid' in enhanced:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(enhanced)
            enhanced_count += 1
        else:
            print(f"  No diagrams matched")
    
    print(f"\n✓ Enhanced {enhanced_count} files with diagrams")

if __name__ == '__main__':
    main()
