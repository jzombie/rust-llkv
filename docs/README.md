# DeepWiki Markdown Extractor

A generic utility for extracting wiki documentation from [DeepWiki.com](https://deepwiki.com) as individual markdown files via web scraping.

## Features

- Scrapes wiki pages directly from DeepWiki website
- Converts HTML to clean markdown using html2text
- Extracts all main wiki pages as separate markdown files
- Preserves page hierarchy and numbering
- Supports any GitHub repository indexed by DeepWiki
- Self-contained Docker image (Python-based)
- No authentication required

## Prerequisites

- Docker installed on your system
- Internet connection (to access DeepWiki MCP server)

## Usage

### With Docker (Recommended)

1. **Build the Docker image:**
   ```bash
   docker build -t deepwiki-scraper docs
   ```

2. **Run the scraper:**
   ```bash
   docker run --rm \
     -v "$PWD/output:/workspace/output" \
     deepwiki-scraper \
     python /usr/local/bin/deepwiki-scraper.py <owner/repo> /workspace/output
   ```

   **Example:**
   ```bash
   docker run --rm \
     -v "$PWD/docs/wiki:/workspace/output" \
     deepwiki-scraper \
     python /usr/local/bin/deepwiki-scraper.py jzombie/rust-llkv /workspace/output
   ```

### With Python (Local)

If you have Python 3.11+ installed:

1. **Install dependencies:**
   ```bash
   pip install -r docs/tools/requirements.txt
   ```

2. **Run the scraper:**
   ```bash
   python docs/tools/deepwiki-scraper.py <owner/repo> <output-directory>
   ```

   **Example:**
   ```bash
   python docs/tools/deepwiki-scraper.py jzombie/rust-llkv ./wiki-output
   ```

## Output Format

The extractor creates individual markdown files with the naming convention:

```
<page-number>-<page-title>.md
```

**Examples:**
- `1-overview.md`
- `2-1-workspace-and-crates.md`
- `3-2-sql-parser.md`

Each file contains:
- The page title as an H1 header
- Full markdown content from DeepWiki
- Proper formatting and structure

## How It Works

1. **Fetch Main Page:** Connects to DeepWiki website and fetches the repository's wiki home page
2. **Extract Navigation:** Parses HTML to find all wiki page links in the navigation
3. **Scrape Each Page:** Fetches each wiki page's HTML content
4. **Convert to Markdown:** Uses html2text to convert HTML to clean markdown
5. **Save Files:** Writes individual markdown files with clean filenames

## Technical Details

- **Scraping:** Direct HTTP requests to `https://deepwiki.com/<owner>/<repo>`
- **HTML Parsing:** BeautifulSoup4 for robust HTML parsing
- **Markdown Conversion:** html2text for HTML to Markdown conversion
- **Dependencies:** requests, beautifulsoup4, html2text (see `tools/requirements.txt`)
- **Rate Limiting:** 1 second delay between requests to be respectful

## Troubleshooting

### "No wiki pages found"
The repository may not be indexed by DeepWiki, or the HTML structure has changed. Try visiting `https://deepwiki.com/<owner>/<repo>` in a browser to verify the wiki exists.

### "Could not find content on page"
The HTML structure of DeepWiki may have changed. The scraper looks for common content selectors (`article`, `main`, `.wiki-content`, etc.) and may need updating.

### Connection timeouts
The scraper includes automatic retries (3 attempts per page). If issues persist, check your internet connection or try again later.

## Example: Extracting LLKV Documentation

To extract the LLKV project documentation:

```bash
# Build the Docker image
docker build -t deepwiki-scraper docs

# Extract to docs/wiki directory
docker run --rm \
  -v "$PWD/docs/wiki:/workspace/output" \
  deepwiki-scraper \
  python /usr/local/bin/deepwiki-scraper.py jzombie/rust-llkv /workspace/output

# Verify extraction
ls -lh docs/wiki/
```

You should see files like:
```
1-overview.md
2-architecture.md
3-sql-interface.md
4-query-planning.md
...
12-development-guide.md
```

## License

This tool is part of the LLKV project and is provided under the same license terms.

## Credits

- **DeepWiki:** AI-powered documentation service at [deepwiki.com](https://deepwiki.com)
- **BeautifulSoup4:** HTML parsing library
- **html2text:** HTML to Markdown converter
