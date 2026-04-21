#!/bin/bash

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

CMDFILE="${1:-all_commands.txt}"

# Create a log file with a timestamp
LOGFILE="$REPO_ROOT/processing_$(date +'%Y%m%d_%H%M%S').log"

# Loop through each line of the command file
while IFS= read -r line || [[ -n "$line" ]]; do
  echo "Running: $line" | tee -a "$LOGFILE"
  eval "$line" >> "$LOGFILE" 2>&1
done < "$CMDFILE"

echo "Finished all commands at $(date)" | tee -a "$LOGFILE"
