#!/bin/bash

# Default: all_commands.txt. Pass e.g. all_commands.local.txt (from setup_wizard_with_sample).
COMMAND_FILE="${1:-all_commands.txt}"
if [[ ! -f "$COMMAND_FILE" ]]; then
  echo "Command file not found: $COMMAND_FILE" >&2
  exit 1
fi

# Create a log file with a timestamp
LOGFILE="processing_$(date +'%Y%m%d_%H%M%S').log"

# Loop through each line of the command file
while IFS= read -r line || [[ -n "$line" ]]; do
  echo "Running: $line" | tee -a "$LOGFILE"
  eval "$line" >> "$LOGFILE" 2>&1
done < "$COMMAND_FILE"

echo "Finished all commands at $(date)" | tee -a "$LOGFILE"
