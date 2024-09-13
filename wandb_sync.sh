#! /bin/bash

while true; do
    # Check the number of running Slurm jobs for the current user
    NUM_JOBS=$(squeue | grep -c " R ")
    date

    if [ "$NUM_JOBS" -eq "0" ]; then
        echo "No running jobs found. Ours is probably finished."
        break
    else
        echo "syncing wandb..."
        poetry run wandb sync --sync-all
        sleep 300
    fi
done
