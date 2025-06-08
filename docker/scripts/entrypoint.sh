#!/bin/bash

# Entrypoint script for Video Advertisement Placement System
# Handles proper initialization, signal handling, and graceful shutdown

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1" >&2
}

log_warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS:${NC} $1"
}

# Global variables
PID=0
SHUTDOWN_INITIATED=false

# Signal handler for graceful shutdown
signal_handler() {
    if [ "$SHUTDOWN_INITIATED" = false ]; then
        SHUTDOWN_INITIATED=true
        log "Received shutdown signal, initiating graceful shutdown..."
        
        if [ $PID -ne 0 ]; then
            log "Sending SIGTERM to process $PID"
            kill -TERM "$PID" 2>/dev/null || true
            
            # Wait for process to terminate gracefully
            local timeout=30
            local count=0
            while kill -0 "$PID" 2>/dev/null && [ $count -lt $timeout ]; do
                sleep 1
                count=$((count + 1))
                log "Waiting for graceful shutdown... ($count/$timeout)"
            done
            
            # Force kill if still running
            if kill -0 "$PID" 2>/dev/null; then
                log_warn "Process did not terminate gracefully, forcing shutdown"
                kill -KILL "$PID" 2>/dev/null || true
            fi
        fi
        
        cleanup
        log_success "Shutdown completed"
        exit 0
    fi
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Clean up temporary files
    if [ -d "$TMP_DIR" ]; then
        find "$TMP_DIR" -type f -name "*.tmp" -mtime +1 -delete 2>/dev/null || true
        find "$TMP_DIR" -type f -name "*.lock" -delete 2>/dev/null || true
    fi
    
    # Clean up GPU memory if PyTorch is available
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('GPU memory cache cleared')
except:
    pass
" 2>/dev/null || true
    
    log "Cleanup completed"
}

# Initialize environment
initialize_environment() {
    log "Initializing environment..."
    
    # Set default environment variables if not provided
    export ENVIRONMENT=${ENVIRONMENT:-production}
    export LOG_LEVEL=${LOG_LEVEL:-INFO}
    export MODEL_CACHE_DIR=${MODEL_CACHE_DIR:-/models}
    export DATA_DIR=${DATA_DIR:-/data}
    export CACHE_DIR=${CACHE_DIR:-/cache}
    export TMP_DIR=${TMP_DIR:-/tmp/video_processing}
    
    # Create necessary directories
    mkdir -p "$MODEL_CACHE_DIR" "$DATA_DIR" "$CACHE_DIR" "$TMP_DIR" /app/logs
    
    # Set proper permissions
    chmod 755 "$MODEL_CACHE_DIR" "$DATA_DIR" "$CACHE_DIR" "$TMP_DIR" /app/logs
    
    log_success "Environment initialized"
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            log_success "NVIDIA GPU detected and accessible"
        else
            log_error "NVIDIA GPU detected but not accessible"
            return 1
        fi
    else
        log_warn "nvidia-smi not found, GPU may not be available"
    fi
    
    # Check CUDA
    if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
        log_success "PyTorch CUDA support verified"
    else
        log_error "PyTorch CUDA support not available"
        return 1
    fi
    
    # Check disk space
    local required_space_gb=5
    local available_space=$(df "$DATA_DIR" | tail -1 | awk '{print $4}')
    local available_space_gb=$((available_space / 1024 / 1024))
    
    if [ $available_space_gb -lt $required_space_gb ]; then
        log_error "Insufficient disk space: ${available_space_gb}GB available, ${required_space_gb}GB required"
        return 1
    fi
    
    log_success "System requirements check passed"
}

# Wait for dependencies
wait_for_dependencies() {
    log "Waiting for dependencies..."
    
    # Wait for Redis if configured
    if [ -n "$REDIS_URL" ] || [ -n "$CELERY_BROKER_URL" ]; then
        local redis_host
        local redis_port
        
        if [ -n "$REDIS_URL" ]; then
            redis_host=$(echo "$REDIS_URL" | sed 's/redis:\/\/\([^:]*\).*/\1/')
            redis_port=$(echo "$REDIS_URL" | sed 's/.*:\([0-9]*\).*/\1/')
        else
            redis_host=$(echo "$CELERY_BROKER_URL" | sed 's/redis:\/\/\([^:]*\).*/\1/')
            redis_port=$(echo "$CELERY_BROKER_URL" | sed 's/.*:\([0-9]*\).*/\1/')
        fi
        
        redis_port=${redis_port:-6379}
        
        log "Waiting for Redis at $redis_host:$redis_port..."
        timeout=60
        count=0
        while ! nc -z "$redis_host" "$redis_port" 2>/dev/null && [ $count -lt $timeout ]; do
            sleep 1
            count=$((count + 1))
        done
        
        if [ $count -ge $timeout ]; then
            log_error "Redis connection timeout"
            return 1
        fi
        
        log_success "Redis connection established"
    fi
    
    # Wait for PostgreSQL if configured
    if [ -n "$DATABASE_URL" ]; then
        local db_host
        local db_port
        
        db_host=$(echo "$DATABASE_URL" | sed 's/.*@\([^:]*\).*/\1/')
        db_port=$(echo "$DATABASE_URL" | sed 's/.*:\([0-9]*\)\/.*/\1/')
        db_port=${db_port:-5432}
        
        log "Waiting for PostgreSQL at $db_host:$db_port..."
        timeout=60
        count=0
        while ! nc -z "$db_host" "$db_port" 2>/dev/null && [ $count -lt $timeout ]; do
            sleep 1
            count=$((count + 1))
        done
        
        if [ $count -ge $timeout ]; then
            log_error "PostgreSQL connection timeout"
            return 1
        fi
        
        log_success "PostgreSQL connection established"
    fi
}

# Preload models if needed
preload_models() {
    if [ "$PRELOAD_MODELS" = "true" ]; then
        log "Preloading models..."
        
        python3 -c "
try:
    from video_ad_placement.core.depth_estimation import DepthEstimator, DepthEstimationConfig
    from video_ad_placement.core.object_detection import ObjectDetector, ObjectDetectionConfig
    
    # Initialize components to trigger model downloads/loading
    depth_config = DepthEstimationConfig()
    depth_estimator = DepthEstimator(depth_config)
    
    detection_config = ObjectDetectionConfig()
    object_detector = ObjectDetector(detection_config)
    
    print('Models preloaded successfully')
except Exception as e:
    print(f'Model preloading failed: {e}')
    exit(1)
"
        
        if [ $? -eq 0 ]; then
            log_success "Models preloaded successfully"
        else
            log_error "Model preloading failed"
            return 1
        fi
    fi
}

# Main execution
main() {
    log "Starting Video Advertisement Placement System..."
    
    # Set up signal handlers
    trap signal_handler SIGTERM SIGINT SIGHUP SIGQUIT
    
    # Initialize environment
    initialize_environment
    
    # Check requirements
    if ! check_requirements; then
        log_error "System requirements not met"
        exit 1
    fi
    
    # Wait for dependencies
    if ! wait_for_dependencies; then
        log_error "Dependencies not available"
        exit 1
    fi
    
    # Preload models if requested
    if ! preload_models; then
        log_error "Model preloading failed"
        exit 1
    fi
    
    # Run database migrations if needed
    if [ -n "$DATABASE_URL" ] && [ "$RUN_MIGRATIONS" = "true" ]; then
        log "Running database migrations..."
        python3 -m video_ad_placement.db.migrate || {
            log_error "Database migrations failed"
            exit 1
        }
    fi
    
    log_success "Initialization completed, starting main process..."
    
    # Execute the main command
    if [ $# -eq 0 ]; then
        log_error "No command provided"
        exit 1
    fi
    
    # Start the main process in background
    "$@" &
    PID=$!
    
    log "Main process started with PID: $PID"
    
    # Wait for the process to complete
    wait $PID
    local exit_code=$?
    
    if [ $exit_code -ne 0 ] && [ "$SHUTDOWN_INITIATED" = false ]; then
        log_error "Main process exited with code $exit_code"
    fi
    
    cleanup
    exit $exit_code
}

# Run main function with all arguments
main "$@" 