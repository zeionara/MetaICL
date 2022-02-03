#!/bin/bash
set -x

# Initialize sweep
SWEEP_CAPTURE=$(wandb sweep -e junshern -p metaicl sweep.yaml 2>&1) # Initialize sweep and capture the terminal output
SWEEP_ID=$(echo $SWEEP_CAPTURE | awk '{print $NF}') # Grab the ID from the last line of the output
echo "Created sweep: $SWEEP_ID"

# Run jobs
echo "Launching 10 workers for the sweep."
sbatch launch_sweep_agents.sbatch $SWEEP_ID