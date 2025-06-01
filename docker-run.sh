#!/bin/bash

# PDG MCP Server Docker Management Script
# This script provides easy commands to build, run, and manage the PDG MCP server in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Default values
IMAGE_NAME="pdg-mcp-server"
CONTAINER_NAME="pdg-mcp-server"
TAG="latest"

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_warning "docker-compose not found. Using docker compose instead."
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
}

# Build the Docker image
build_image() {
    print_header "Building PDG MCP Server Docker Image"
    
    print_info "Building image: $IMAGE_NAME:$TAG"
    
    if docker build -t "$IMAGE_NAME:$TAG" .; then
        print_success "Docker image built successfully"
        
        # Show image info
        echo ""
        print_info "Image details:"
        docker images "$IMAGE_NAME:$TAG" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Run the container
run_server() {
    print_header "Running PDG MCP Server"
    
    # Create data directory if it doesn't exist
    mkdir -p ./pdg_data
    
    print_info "Starting container: $CONTAINER_NAME"
    
    if $COMPOSE_CMD up -d pdg-mcp-server; then
        print_success "PDG MCP Server started successfully"
        
        # Show container status
        echo ""
        print_info "Container status:"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        
        echo ""
        print_info "View logs with: $0 logs"
        print_info "Stop server with: $0 stop"
    else
        print_error "Failed to start PDG MCP Server"
        exit 1
    fi
}

# Run tests in container
run_tests() {
    print_header "Running PDG MCP Server Tests"
    
    mkdir -p ./pdg_data
    
    if $COMPOSE_CMD --profile testing up pdg-mcp-test; then
        print_success "Tests completed"
    else
        print_warning "Some tests may have failed"
    fi
    
    # Clean up test container
    $COMPOSE_CMD --profile testing down
}

# Run examples in container
run_examples() {
    print_header "Running PDG MCP Server Examples"
    
    mkdir -p ./pdg_data
    
    print_info "Starting interactive examples session..."
    print_warning "Press Ctrl+C to exit when done"
    
    $COMPOSE_CMD --profile examples up pdg-mcp-examples
    
    # Clean up examples container
    $COMPOSE_CMD --profile examples down
}

# Show logs
show_logs() {
    print_header "PDG MCP Server Logs"
    
    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
        docker logs -f "$CONTAINER_NAME"
    else
        print_error "Container $CONTAINER_NAME is not running"
        exit 1
    fi
}

# Stop the server
stop_server() {
    print_header "Stopping PDG MCP Server"
    
    if $COMPOSE_CMD down; then
        print_success "PDG MCP Server stopped"
    else
        print_warning "Server may not have been running"
    fi
}

# Clean up Docker resources
cleanup() {
    print_header "Cleaning Up Docker Resources"
    
    print_info "Stopping and removing containers..."
    $COMPOSE_CMD down --remove-orphans
    
    print_info "Removing Docker image..."
    if docker rmi "$IMAGE_NAME:$TAG" 2>/dev/null; then
        print_success "Docker image removed"
    else
        print_warning "Docker image may not exist"
    fi
    
    print_info "Removing Docker volumes..."
    if docker volume rm "$(basename $PWD)_pdg_data" 2>/dev/null; then
        print_success "Docker volumes removed"
    else
        print_warning "Docker volumes may not exist"
    fi
    
    print_success "Cleanup completed"
}

# Show container status
status() {
    print_header "PDG MCP Server Status"
    
    echo "Container Status:"
    docker ps -a --filter "name=pdg-mcp" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.CreatedSince}}"
    
    echo ""
    echo "Image Status:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
    
    echo ""
    echo "Volume Status:"
    docker volume ls --filter "name=pdg_data" --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"
}

# Interactive shell in container
shell() {
    print_header "Opening Interactive Shell"
    
    if docker ps --filter "name=$CONTAINER_NAME" --format "{{.Names}}" | grep -q "$CONTAINER_NAME"; then
        print_info "Connecting to running container..."
        docker exec -it "$CONTAINER_NAME" bash
    else
        print_info "Starting new container with shell..."
        docker run -it --rm \
            -v "$(pwd)/pdg_data:/app/data" \
            "$IMAGE_NAME:$TAG" bash
    fi
}

# Show help
show_help() {
    echo "PDG MCP Server Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  run         Start the PDG MCP server"
    echo "  stop        Stop the PDG MCP server"
    echo "  restart     Restart the PDG MCP server"
    echo "  test        Run the test suite"
    echo "  examples    Run interactive examples"
    echo "  logs        Show server logs (follow mode)"
    echo "  status      Show container and image status"
    echo "  shell       Open interactive shell in container"
    echo "  cleanup     Remove all Docker resources"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build           # Build the image"
    echo "  $0 run             # Start the server"
    echo "  $0 logs            # Watch server logs"
    echo "  $0 test            # Run tests"
    echo "  $0 shell           # Open bash shell"
    echo ""
}

# Main script logic
main() {
    check_docker
    
    case "${1:-help}" in
        build)
            build_image
            ;;
        run)
            run_server
            ;;
        stop)
            stop_server
            ;;
        restart)
            stop_server
            sleep 2
            run_server
            ;;
        test)
            run_tests
            ;;
        examples)
            run_examples
            ;;
        logs)
            show_logs
            ;;
        status)
            status
            ;;
        shell)
            shell
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@" 