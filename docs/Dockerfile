FROM python:3.11-slim as scraper

WORKDIR /workspace

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies using uv
COPY tools/requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

# Copy the scraper script
COPY tools/deepwiki-scraper.py /usr/local/bin/deepwiki-scraper.py
RUN chmod +x /usr/local/bin/deepwiki-scraper.py

# Stage 2: Build mdbook tools
FROM rust:latest as builder

# Install mdbook and mdbook-mermaid using cargo
RUN cargo install mdbook mdbook-mermaid

# Stage 3: Final image with both scraper and mdbook
FROM python:3.11-slim

WORKDIR /workspace

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python dependencies
COPY tools/requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

# Copy mdbook binaries from builder (cargo installs to /usr/local/cargo/bin)
COPY --from=builder /usr/local/cargo/bin/mdbook /usr/local/bin/
COPY --from=builder /usr/local/cargo/bin/mdbook-mermaid /usr/local/bin/

# Copy the scraper script
COPY tools/deepwiki-scraper.py /usr/local/bin/deepwiki-scraper.py
RUN chmod +x /usr/local/bin/deepwiki-scraper.py

# Copy build script
COPY build-docs.sh /usr/local/bin/build-docs.sh
RUN chmod +x /usr/local/bin/build-docs.sh

# Default command builds everything
CMD ["/usr/local/bin/build-docs.sh"]
