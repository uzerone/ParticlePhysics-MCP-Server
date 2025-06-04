#!/bin/bash

# PDG MCP Server Docker Convenience Script
# Usage: ./docker-run.sh [build|run|test|examples|shell|clean]

set -e

IMAGE_NAME="pdg-mcp-server"
CONTAINER_NAME="pdg-mcp-server"

case "$1" in
    build)
        echo "Building PDG MCP Server Docker image..."
        docker build -t $IMAGE_NAME:latest .
        echo "✅ Build complete!"
        ;;
    
    run)
        echo "Running PDG MCP Server..."
        docker run --rm --name $CONTAINER_NAME \
            -v "$(pwd)/pdg_data:/app/data" \
            $IMAGE_NAME:latest
        ;;
    
    test)
        echo "Running tests in Docker container..."
        docker run --rm $IMAGE_NAME:latest python test_modular.py
        ;;
    
    examples)
        echo "Running examples in Docker container..."
        docker run --rm -it $IMAGE_NAME:latest python examples.py
        ;;
    
    shell)
        echo "Opening shell in Docker container..."
        docker run --rm -it \
            -v "$(pwd)/pdg_data:/app/data" \
            $IMAGE_NAME:latest bash
        ;;
    
    cli)
        shift
        echo "Running CLI command: $@"
        docker run --rm $IMAGE_NAME:latest python pdg_cli.py "$@"
        ;;
    
    clean)
        echo "Cleaning up Docker containers and images..."
        docker container prune -f
        docker image rm $IMAGE_NAME:latest 2>/dev/null || true
        echo "✅ Cleanup complete!"
        ;;
    
    compose-up)
        echo "Starting with Docker Compose..."
        docker-compose up --build
        ;;
    
    compose-down)
        echo "Stopping Docker Compose..."
        docker-compose down
        ;;
    
    *)
        echo "PDG MCP Server Docker Convenience Script"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  build        Build the Docker image"
        echo "  run          Run the MCP server"
        echo "  test         Run tests in container"
        echo "  examples     Run examples interactively"
        echo "  shell        Open bash shell in container"
        echo "  cli [args]   Run CLI command with arguments"
        echo "  clean        Clean up containers and images"
        echo "  compose-up   Start with Docker Compose"
        echo "  compose-down Stop Docker Compose"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 test"
        echo "  $0 cli search --query 'e-'"
        echo "  $0 examples"
        ;;
esac 