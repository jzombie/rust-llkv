#!/usr/bin/env python3
import html2text

html = """<h3>Layer Diagram</h3>
<p>The following diagram shows the system layers:</p>
<pre><code class="language-mermaid">```mermaid
graph TB
    A --> B
```</code></pre>
<p>Additional content after diagram.</p>"""

def convert_html_to_markdown(html_content):
    """Convert HTML to markdown using html2text and preserve mermaid diagrams"""
    # Use html2text for conversion
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0  # No line wrapping
    markdown = h.handle(html_content)
    
    # Post-process: Convert indented mermaid blocks to fenced blocks
    # html2text converts <pre><code class="language-mermaid"> to indented blocks
    # The format is:
    #     ```mermaid
    #     graph TB
    #     ...
    #     ```
    # We need to detect these and convert to proper fenced blocks
    
    lines = markdown.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line starts with "    ```mermaid" (indented opening fence)
        if line.strip() == '```mermaid' and (i == 0 or not lines[i-1].startswith('```')):
            # Found an indented mermaid block opener
            # Skip this line and collect until we find the closing ```
            i += 1
            diagram_lines = []
            
            while i < len(lines):
                current = lines[i]
                stripped = current.strip()
                
                # Check for closing fence
                if stripped == '```':
                    i += 1
                    break
                    
                # Remove 4-space indent if present
                if current.startswith('    '):
                    diagram_lines.append(current[4:])
                else:
                    diagram_lines.append(current)
                i += 1
            
            # Output as fenced block (without indentation)
            result.append('```mermaid')
            result.extend(diagram_lines)
            result.append('```')
            result.append('')  # Add blank line after
            continue
        
        result.append(line)
        i += 1
    
    return '\n'.join(result).strip()

result = convert_html_to_markdown(html)
print("=== CONVERTED OUTPUT ===")
print(result)
print("\n=== CHECKING FOR FENCED MERMAID ===")
if '```mermaid\ngraph TB' in result:
    print("✓ Found properly formatted mermaid block!")
else:
    print("✗ No properly formatted mermaid block found")
    print("Looking for patterns:")
    print("  - '```mermaid' count:", result.count('```mermaid'))
    print("  - 'graph TB' count:", result.count('graph TB'))
