# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set metadata
LABEL maintainer="PDG MCP Server"
LABEL description="Containerized PDG (Particle Data Group) MCP Server for particle physics research"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PDG_MCP_HOME=/app
ENV PDG_DATA_DIR=/app/data

# Create non-root user for security
RUN groupadd -r pdguser && useradd -r -g pdguser pdguser

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR $PDG_MCP_HOME

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY pdg_mcp_server.py .
COPY pdg_cli.py .
COPY test_modular.py .
COPY examples.py .
COPY setup.py .
COPY README.md .
COPY LICENSE .
COPY pdg_modules/ ./pdg_modules/

# Create data directory for PDG database
RUN mkdir -p $PDG_DATA_DIR

# Make scripts executable
RUN chmod +x pdg_mcp_server.py pdg_cli.py test_modular.py examples.py

# Change ownership to non-root user
RUN chown -R pdguser:pdguser $PDG_MCP_HOME

# Switch to non-root user
USER pdguser

# Test the installation during build
RUN python -c "import mcp.types; print('✓ MCP package installed')"
RUN python -c "import pdg; api = pdg.connect(); print('✓ PDG package working')"

# Expose port (although MCP typically uses stdin/stdout)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import pdg; pdg.connect()" || exit 1

# Default command
CMD ["python", "pdg_mcp_server.py"]

# Alternative commands can be:
# CMD ["python", "test_modular.py"]     # Run tests
# CMD ["python", "examples.py"]         # Run examples
# CMD ["bash"]                          # Interactive shell 