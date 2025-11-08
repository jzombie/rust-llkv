# DeepWiki Documentation Builder

A generic utility for extracting wiki documentation from [DeepWiki.com](https://deepwiki.com) and building it into beautiful HTML documentation with mdBook.

## Features

- Scrapes wiki pages directly from DeepWiki website
- Converts HTML to clean markdown using html2text
- Extracts mermaid diagrams from JavaScript payload using intelligent fuzzy matching
- Preserves page hierarchy and numbering
- Builds beautiful HTML documentation with mdBook
- Supports any GitHub repository indexed by DeepWiki
- Fully configurable via environment variables
- Self-contained Docker image
- No authentication required

## Prerequisites

- Docker installed on your system
- Internet connection (to access DeepWiki)

## Usage

### Quick Start

Build the complete documentation (markdown + HTML book):

```bash
# Build the Docker image
cd docs
docker build -t deepwiki-scraper .

# Run the complete build
docker run --rm \
  -e REPO="owner/repo" \
  -e BOOK_TITLE="My Documentation" \
  -v "$(pwd)/output:/output" \
  deepwiki-scraper
```

### Configuration Options

All configuration is done via environment variables:

| Variable            | Description                                  | Default                       |
| ------------------- | -------------------------------------------- | ----------------------------- |
| `REPO`              | GitHub repository (owner/repo)               | `jzombie/rust-llkv`           |
| `BOOK_TITLE`        | Title for the documentation book             | `Documentation`               |
| `BOOK_AUTHORS`      | Author name(s)                               | Auto-detected from repo owner |
| `GIT_REPO_URL`      | Git repository URL                           | Auto-constructed from REPO    |
| `EDIT_URL_TEMPLATE` | Edit URL template for "Edit this page" links | Auto-constructed              |
| `MARKDOWN_ONLY`     | Skip mdBook build, only extract markdown     | `false`                       |

### Markdown-Only Mode (Debugging)

For faster iteration when debugging or just extracting markdown:

```bash
docker run --rm \
  -e REPO="owner/repo" \
  -e MARKDOWN_ONLY="true" \
  -v "$(pwd)/output:/output" \
  deepwiki-scraper
```

This will:
- Scrape wiki pages from DeepWiki
- Extract and intelligently place mermaid diagrams
- Output only the markdown files (skip mdBook build)
- Much faster for debugging diagram placement or content extraction

### Complete Example

```bash
# Build documentation for LLKV project
docker run --rm \
  -e REPO="jzombie/rust-llkv" \
  -e BOOK_TITLE="LLKV Documentation" \
  -v "$(pwd)/output:/output" \
  deepwiki-scraper

# Serve the documentation locally
cd output
python3 -m http.server --directory book 8000
# Open http://localhost:8000 in your browser
```

## Output Format

### Full Build Mode

When built without `MARKDOWN_ONLY=true`, you get:

- `output/book/` - Complete HTML documentation
  - Searchable, responsive mdBook site
  - Working mermaid diagram rendering
  - Navigation sidebar with hierarchy
  - "Edit this page" links to GitHub
- `output/markdown/` - Source markdown files
  - Individual markdown files with naming: `<number>-<title>.md`
  - Subsections in `section-N/` directories
  - Enhanced with intelligently-placed mermaid diagrams
- `output/book.toml` - mdBook configuration file

### Markdown-Only Mode

When built with `MARKDOWN_ONLY=true`:

- `output/markdown/` - Source markdown files only
  - Same structure as above
  - Faster for debugging diagram placement
  - Use when you only need the markdown content

**Example filenames:**
- `1-overview.md`
- `2-1-workspace-and-crates.md`
- `3-2-sql-parser.md`
- `section-4/4-1-logical-planning.md`

## How It Works

### Phase 1: Clean Markdown Extraction
1. Fetches wiki structure from DeepWiki
2. Scrapes each page's HTML content
3. Removes navigation and UI elements
4. Converts to clean markdown using html2text
5. Saves to temporary directory

### Phase 2: Diagram Enhancement
1. Extracts JavaScript payload from any DeepWiki page
2. Finds ~461 total mermaid diagrams embedded in JavaScript
3. Extracts ~48 diagrams with sufficient context (500-char snippets)
4. Uses fuzzy matching with progressive chunk sizes (300→200→150→100→80 chars)
5. Intelligently inserts diagrams after relevant paragraphs
6. Moves completed files atomically to output directory

### Phase 3: mdBook Build (unless MARKDOWN_ONLY=true)
1. Initializes mdBook structure with configuration
2. Auto-generates SUMMARY.md table of contents from file structure
3. Copies markdown files to book source
4. Installs mdbook-mermaid assets (CSS/JS for diagram rendering)
5. Builds complete HTML documentation
6. Copies outputs to /output directory

## Technical Details

- **Scraping:** Direct HTTP requests to DeepWiki
- **HTML Parsing:** BeautifulSoup4 for robust parsing
- **Markdown Conversion:** html2text with body_width=0
- **Diagram Extraction:** Regex pattern matching on JavaScript `self.__next_f.push` calls
- **Fuzzy Matching:** Normalized whitespace, progressive chunk comparison, scoring system
- **Documentation Build:** mdBook with rust theme + mdbook-mermaid plugin
- **Package Management:** uv (modern Python package manager)
- **Dependencies:** Python 3.11, Rust (latest), mdbook, mdbook-mermaid
- **Architecture:** Multi-stage Docker build (Rust builder + Python runtime)

## Troubleshooting

### "No wiki pages found"
The repository may not be indexed by DeepWiki. Try visiting `https://deepwiki.com/<owner>/<repo>` in a browser to verify the wiki exists.

### "Could not find content on page"
The HTML structure of DeepWiki may have changed. The scraper looks for common content selectors and may need updating.

### Diagrams not appearing
- Check that the wiki has mermaid diagrams on DeepWiki's website
- Use `MARKDOWN_ONLY=true` to debug the markdown output
- Diagrams are matched using fuzzy matching - some may not have enough context to match accurately

### Connection timeouts
The scraper includes automatic retries (3 attempts per page). If issues persist, check your internet connection.

### mdBook build fails
- Ensure Docker has enough memory (2GB+ recommended)
- Check that the Rust toolchain installed correctly in the image
- Try `MARKDOWN_ONLY=true` to verify markdown extraction works independently

## Examples

### Extract markdown only (fast debugging)
```bash
docker run --rm \
  -e REPO="facebook/react" \
  -e MARKDOWN_ONLY="true" \
  -v "$(pwd)/output:/output" \
  deepwiki-scraper
```

### Build complete documentation
```bash
docker run --rm \
  -e REPO="facebook/react" \
  -e BOOK_TITLE="React Documentation" \
  -e BOOK_AUTHORS="Meta Open Source" \
  -v "$(pwd)/output:/output" \
  deepwiki-scraper
```

### Use with any DeepWiki repository
```bash
docker run --rm \
  -e REPO="microsoft/vscode" \
  -e BOOK_TITLE="VS Code Internals" \
  -v "$(pwd)/vscode-docs:/output" \
  deepwiki-scraper
```

## License

This tool is part of the LLKV project and is provided under the same license terms.

## Credits

- **DeepWiki:** AI-powered documentation service at [deepwiki.com](https://deepwiki.com)
- **mdBook:** Rust-based documentation builder
- **mdbook-mermaid:** Mermaid diagram support for mdBook
- **BeautifulSoup4:** HTML parsing library
- **html2text:** HTML to Markdown converter
- **uv:** Fast Python package installer

## Architecture Notes

This tool is designed to be fully generic and can be extracted as a standalone package:

- **No hardcoded repository specifics** - all configuration via environment variables
- **Dynamic structure discovery** - auto-generates table of contents from actual files
- **Fuzzy diagram matching** - works with DeepWiki's client-side rendering
- **Temp directory workflow** - atomic operations, no partial states
- **Multi-stage Docker build** - optimized image size with both Python and Rust tools

Perfect for:
- Extracting any DeepWiki wiki as markdown
- Building searchable HTML documentation
- Creating offline documentation archives
- Integrating into CI/CD pipelines
