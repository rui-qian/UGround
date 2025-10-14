#!/bin/bash

# Function to check if port is available
is_port_available() {
    local port=$1
    if command -v ss >/dev/null 2>&1; then
        ! ss -tuln | grep -q ":${port} "
    elif command -v netstat >/dev/null 2>&1; then
        ! netstat -tuln 2>/dev/null | grep -q ":${port} "
    else
        return 0  # Assume available if no tools
    fi
}

# Find next available port starting from base
find_next_port() {
    local start_port=${1:-24994}
    local port=$start_port
    
    while [ $port -lt $((start_port + 1000)) ]; do
        if is_port_available $port; then
            echo $port
            return 0
        fi
        port=$((port + 1))
    done
    
    echo $start_port  # Fallback
}

# Main logic
if [ -z "$MASTER_PORT" ]; then
    MASTER_PORT=$((24900 + RANDOM % 91))
fi

if ! is_port_available $MASTER_PORT; then
    MASTER_PORT=$(find_next_port $MASTER_PORT)
    echo "Port conflict detected, using port: $MASTER_PORT"
fi

export MASTER_PORT 