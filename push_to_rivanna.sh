#!/bin/bash
# Script to push code to Rivanna HPC
# Usage: ./push_to_rivanna.sh [config|code|all]

USERNAME="njt4xc"
SERVER="login.hpc.virginia.edu"
REMOTE_DIR="~/SelfMedMAE"

if [ "$1" == "config" ]; then
    echo "Pushing config files..."
    scp configs/*.yaml ${USERNAME}@${SERVER}:${REMOTE_DIR}/configs/
elif [ "$1" == "code" ]; then
    echo "Pushing code files..."
    scp -r lib/ ${USERNAME}@${SERVER}:${REMOTE_DIR}/
    scp main.py ${USERNAME}@${SERVER}:${REMOTE_DIR}/
    scp configs/*.yaml ${USERNAME}@${SERVER}:${REMOTE_DIR}/configs/
elif [ "$1" == "all" ]; then
    echo "Pushing everything..."
    scp -r * ${USERNAME}@${SERVER}:${REMOTE_DIR}/
else
    echo "Usage: $0 [config|code|all]"
    echo "  config - Push only config files"
    echo "  code   - Push code files (lib/, main.py, configs/)"
    echo "  all    - Push everything"
    exit 1
fi

echo "Done!"

