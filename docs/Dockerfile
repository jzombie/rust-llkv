FROM python:3.11-slim

WORKDIR /workspace

# Install dependencies
COPY tools/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the scraper script
COPY tools/deepwiki-scraper.py /usr/local/bin/deepwiki-scraper.py
RUN chmod +x /usr/local/bin/deepwiki-scraper.py

# Default command shows usage
CMD ["python", "/usr/local/bin/deepwiki-scraper.py"]
