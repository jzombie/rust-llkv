#!/bin/bash
set -e

echo "================================================================================"
echo "DeepWiki Documentation Builder"
echo "================================================================================"

# Configuration - all can be overridden via environment variables
REPO="${REPO:-jzombie/rust-llkv}"
BOOK_TITLE="${BOOK_TITLE:-Documentation}"
BOOK_AUTHORS="${BOOK_AUTHORS:-}"
GIT_REPO_URL="${GIT_REPO_URL:-}"
EDIT_URL_TEMPLATE="${EDIT_URL_TEMPLATE:-}"
WORK_DIR="/workspace"
WIKI_DIR="$WORK_DIR/wiki"
OUTPUT_DIR="/output"
BOOK_DIR="$WORK_DIR/book"

# Extract repo parts for defaults
REPO_OWNER=$(echo "$REPO" | cut -d'/' -f1)
REPO_NAME=$(echo "$REPO" | cut -d'/' -f2)

# Set defaults if not provided
: "${BOOK_AUTHORS:=$REPO_OWNER}"
: "${GIT_REPO_URL:=https://github.com/$REPO}"
: "${EDIT_URL_TEMPLATE:=https://github.com/$REPO/edit/main/docs/wiki/{path}}"

echo ""
echo "Configuration:"
echo "  Repository:    $REPO"
echo "  Book Title:    $BOOK_TITLE"
echo "  Authors:       $BOOK_AUTHORS"
echo "  Git Repo URL:  $GIT_REPO_URL"

# Step 1: Scrape wiki
echo ""
echo "Step 1: Scraping wiki from DeepWiki..."
python3 /usr/local/bin/deepwiki-scraper.py "$REPO" "$WIKI_DIR"

# Step 2: Initialize mdbook structure
echo ""
echo "Step 2: Initializing mdBook structure..."
mkdir -p "$BOOK_DIR"
cd "$BOOK_DIR"

# Create book.toml
cat > book.toml <<EOF
[book]
title = "$BOOK_TITLE"
authors = ["$BOOK_AUTHORS"]
language = "en"
multilingual = false
src = "src"

[output.html]
default-theme = "rust"
git-repository-url = "$GIT_REPO_URL"
edit-url-template = "$EDIT_URL_TEMPLATE"

[preprocessor.mermaid]
command = "mdbook-mermaid"

[output.html.fold]
enable = true
level = 1
EOF

# Create src directory
mkdir -p src

# Step 3: Generate SUMMARY.md dynamically from scraped files
echo ""
echo "Step 3: Generating SUMMARY.md from scraped content..."

# Generate SUMMARY.md by discovering the actual file structure
{
    echo "# Summary"
    echo ""
    
    # Find the first main page (usually overview/introduction)
    first_page=$(ls "$WIKI_DIR"/*.md 2>/dev/null | head -1 | xargs basename)
    if [ -n "$first_page" ]; then
        title=$(head -1 "$WIKI_DIR/$first_page" | sed 's/^# //')
        echo "[${title:-Introduction}]($first_page)"
        echo ""
    fi
    
    # Process all main pages (files in root, not in section-* directories)
    for file in "$WIKI_DIR"/*.md; do
        [ -f "$file" ] || continue
        filename=$(basename "$file")
        
        # Skip the first page (already added as introduction)
        [ "$filename" = "$first_page" ] && continue
        
        # Extract title from first line of markdown file
        title=$(head -1 "$file" | sed 's/^# //')
        
        # Check if this page has subsections
        section_num=$(echo "$filename" | grep -oE '^[0-9]+' || true)
        section_dir="$WIKI_DIR/section-$section_num"
        
        if [ -n "$section_num" ] && [ -d "$section_dir" ]; then
            # Main section with subsections
            echo "# $title"
            echo ""
            echo "- [$title]($filename)"
            
            # Add subsections
            for subfile in "$section_dir"/*.md; do
                [ -f "$subfile" ] || continue
                subfilename=$(basename "$subfile")
                subtitle=$(head -1 "$subfile" | sed 's/^# //')
                echo "  - [$subtitle](section-$section_num/$subfilename)"
            done
            echo ""
        else
            # Standalone page without subsections
            echo "- [$title]($filename)"
        fi
    done
} > src/SUMMARY.md

echo "Generated SUMMARY.md with $(grep -c '\[' src/SUMMARY.md) entries"

# Step 4: Copy markdown files to book src
echo ""
echo "Step 4: Copying markdown files to book..."
cp -r "$WIKI_DIR"/* src/

# Step 5: Install mermaid support
echo ""
echo "Step 5: Installing mdbook-mermaid assets..."
mdbook-mermaid install "$BOOK_DIR"

# Step 6: Build the book
echo ""
echo "Step 6: Building mdBook..."
mdbook build

# Step 7: Copy outputs
echo ""
echo "Step 7: Copying outputs to /output..."
mkdir -p "$OUTPUT_DIR"

# Copy the built book
cp -r book "$OUTPUT_DIR/"

# Copy the markdown source files
mkdir -p "$OUTPUT_DIR/markdown"
cp -r "$WIKI_DIR"/* "$OUTPUT_DIR/markdown/"

# Copy book configuration for reference
cp book.toml "$OUTPUT_DIR/"

echo ""
echo "================================================================================"
echo "✓ Documentation build complete!"
echo "================================================================================"
echo ""
echo "Outputs:"
echo "  - HTML book:       /output/book/"
echo "  - Markdown files:  /output/markdown/"
echo "  - Book config:     /output/book.toml"
echo ""
echo "To serve the book locally:"
echo "  cd /output && python3 -m http.server --directory book 8000"
echo ""
