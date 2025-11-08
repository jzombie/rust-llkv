FROM python:3.11-slim

WORKDIR /workspace

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies using uv
COPY tools/requirements.txt /tmp/requirements.txt
RUN uv pip install --system --no-cache -r /tmp/requirements.txt

# Copy the scraper script
COPY tools/deepwiki-scraper.py /usr/local/bin/deepwiki-scraper.py
RUN chmod +x /usr/local/bin/deepwiki-scraper.py

# Default command shows usage
CMD ["python", "/usr/local/bin/deepwiki-scraper.py"]
