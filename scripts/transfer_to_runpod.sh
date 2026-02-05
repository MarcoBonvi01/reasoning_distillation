#!/bin/bash
# ============================================================
# Transfer Script for RunPod Cluster
# ============================================================
# This script transfers the trained model and project files
# to a RunPod instance via SSH/SCP/RSYNC
#
# Usage:
#   ./scripts/transfer_to_runpod.sh [command]
#
# Commands:
#   model     - Transfer only the trained model
#   project   - Transfer the entire project (excluding large files)
#   sync      - Sync project with rsync (incremental)
#   download  - Download results from RunPod
#
# ============================================================

# ============================================================
# CONFIGURATION - Edit these values for your RunPod instance
# ============================================================

# RunPod SSH connection details
RUNPOD_HOST="213.173.108.13"          # Your RunPod IP address
RUNPOD_PORT="16416"                    # SSH port from RunPod dashboard
RUNPOD_USER="root"                     # Usually 'root' for RunPod
RUNPOD_KEY="~/.ssh/id_ed25519"        # Path to your SSH key

# Alternative: Use RunPod pod ID (uncomment if using runpodctl)
# RUNPOD_POD_ID="your-pod-id-here"

# Remote paths on RunPod
REMOTE_BASE_DIR="/root/reasoning_distillation"
REMOTE_MODEL_DIR="${REMOTE_BASE_DIR}/experiments/distillation"

# Local paths - ZIP files in Downloads
LOCAL_PROJECT_DIR="$HOME/Downloads/reasoning_distillation"
LOCAL_MODEL_ZIP="$HOME/Downloads/reasoning_distillation/experiments/distillation/best_model.zip"

# ============================================================
# Colors for output
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# Helper functions
# ============================================================

print_header() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_connection() {
    print_header "Testing SSH Connection"
    
    echo "Connecting to ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_PORT}..."
    
    if ssh -i "$RUNPOD_KEY" -p "$RUNPOD_PORT" -o ConnectTimeout=10 \
        "${RUNPOD_USER}@${RUNPOD_HOST}" "echo 'Connection successful'" 2>/dev/null; then
        print_success "SSH connection established"
        return 0
    else
        print_error "Failed to connect to RunPod"
        echo ""
        echo "Please check:"
        echo "  1. RunPod instance is running"
        echo "  2. SSH key is correct: $RUNPOD_KEY"
        echo "  3. Host/Port are correct: ${RUNPOD_HOST}:${RUNPOD_PORT}"
        echo "  4. Your SSH key is added to RunPod"
        return 1
    fi
}

# ============================================================
# Transfer Commands
# ============================================================

transfer_model() {
    print_header "Transferring Trained Model to RunPod"
    
    # Check if model zip exists
    if [ ! -f "$LOCAL_MODEL_ZIP" ]; then
        print_error "Model file not found: $LOCAL_MODEL_ZIP"
        echo "Please check the file exists in Downloads."
        exit 1
    fi
    
    # Show what will be transferred
    echo "Source: $LOCAL_MODEL_ZIP"
    echo "Destination: ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_MODEL_DIR}/"
    echo ""
    
    # Calculate size
    local size=$(du -sh "$LOCAL_MODEL_ZIP" 2>/dev/null | cut -f1)
    echo "Total size: $size"
    echo ""
    
    # Create remote directory
    echo "Creating remote directory..."
    ssh -i "$RUNPOD_KEY" -p "$RUNPOD_PORT" \
        "${RUNPOD_USER}@${RUNPOD_HOST}" \
        "mkdir -p ${REMOTE_MODEL_DIR}"
    
    # Transfer zip file using scp
    echo "Transferring model file..."
    scp -i "$RUNPOD_KEY" -P "$RUNPOD_PORT" \
        "$LOCAL_MODEL_ZIP" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_MODEL_DIR}/"
    
    if [ $? -eq 0 ]; then
        print_success "Model transferred successfully!"
        echo ""
        echo "Now unzipping on RunPod..."
        
        # Unzip on remote
        ssh -i "$RUNPOD_KEY" -p "$RUNPOD_PORT" \
            "${RUNPOD_USER}@${RUNPOD_HOST}" \
            "cd ${REMOTE_MODEL_DIR} && unzip -o best_model.zip && rm best_model.zip"
        
        print_success "Model unzipped and ready at: ${REMOTE_MODEL_DIR}/best_model"
    else
        print_error "Transfer failed"
        exit 1
    fi
}

transfer_project() {
    print_header "Transferring Project to RunPod"
    
    echo "Source: $LOCAL_PROJECT_DIR"
    echo "Destination: ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_BASE_DIR}"
    echo ""
    
    # Create remote directory
    echo "Creating remote directory..."
    ssh -i "$RUNPOD_KEY" -p "$RUNPOD_PORT" \
        "${RUNPOD_USER}@${RUNPOD_HOST}" \
        "mkdir -p ${REMOTE_BASE_DIR}"
    
    # Transfer with exclusions
    echo "Transferring project files (excluding large files)..."
    rsync -avz --progress \
        -e "ssh -i $RUNPOD_KEY -p $RUNPOD_PORT" \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.ipynb_checkpoints' \
        --exclude 'experiments/*/checkpoint-*' \
        --exclude '*.pt' \
        --exclude '*.bin' \
        --exclude '*.safetensors' \
        --exclude 'data/raw/*' \
        --exclude '.venv' \
        --exclude 'venv' \
        --exclude 'node_modules' \
        "$LOCAL_PROJECT_DIR/" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_BASE_DIR}/"
    
    if [ $? -eq 0 ]; then
        print_success "Project transferred successfully!"
        echo ""
        echo "To transfer the trained model as well, run:"
        echo "  ./scripts/transfer_to_runpod.sh model"
    else
        print_error "Transfer failed"
        exit 1
    fi
}

sync_project() {
    print_header "Syncing Project with RunPod (Incremental)"
    
    echo "This will sync only changed files..."
    echo ""
    
    rsync -avz --progress --delete \
        -e "ssh -i $RUNPOD_KEY -p $RUNPOD_PORT" \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.ipynb_checkpoints' \
        --exclude 'experiments/' \
        --exclude 'data/raw/*' \
        --exclude '.venv' \
        --exclude 'venv' \
        "$LOCAL_PROJECT_DIR/" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_BASE_DIR}/"
    
    if [ $? -eq 0 ]; then
        print_success "Sync completed!"
    else
        print_error "Sync failed"
        exit 1
    fi
}

download_results() {
    print_header "Downloading Results from RunPod"
    
    local download_dir="${LOCAL_PROJECT_DIR}/experiments/runpod_results"
    mkdir -p "$download_dir"
    
    echo "Downloading from: ${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_BASE_DIR}/experiments/"
    echo "To: $download_dir"
    echo ""
    
    rsync -avz --progress \
        -e "ssh -i $RUNPOD_KEY -p $RUNPOD_PORT" \
        "${RUNPOD_USER}@${RUNPOD_HOST}:${REMOTE_BASE_DIR}/experiments/" \
        "$download_dir/"
    
    if [ $? -eq 0 ]; then
        print_success "Results downloaded to: $download_dir"
    else
        print_error "Download failed"
        exit 1
    fi
}

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  model     Transfer only the trained model (~300MB for flan-t5-small)"
    echo "  project   Transfer the entire project (excluding large files)"
    echo "  sync      Sync project incrementally (faster for updates)"
    echo "  download  Download experiment results from RunPod"
    echo "  test      Test SSH connection"
    echo "  help      Show this help message"
    echo ""
    echo "Configuration:"
    echo "  Edit the variables at the top of this script to set:"
    echo "  - RUNPOD_HOST: Your RunPod SSH host"
    echo "  - RUNPOD_PORT: SSH port (from RunPod dashboard)"
    echo "  - RUNPOD_KEY:  Path to your SSH private key"
    echo ""
    echo "Example:"
    echo "  # First, transfer the project code"
    echo "  $0 project"
    echo ""
    echo "  # Then, transfer the trained model"
    echo "  $0 model"
    echo ""
    echo "  # After training on RunPod, download results"
    echo "  $0 download"
}

# ============================================================
# Main
# ============================================================

case "${1:-help}" in
    model)
        check_connection && transfer_model
        ;;
    project)
        check_connection && transfer_project
        ;;
    sync)
        check_connection && sync_project
        ;;
    download)
        check_connection && download_results
        ;;
    test)
        check_connection
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
