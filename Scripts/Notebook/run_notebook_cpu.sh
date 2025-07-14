#!/bin/bash
# Generate a random port number between 49152 and 59151
MYPORT=$(($(($RANDOM % 10000))+49152))

# Echo the port number for the user to see
echo "Using port: $MYPORT"

# Run the jupyter-notebook command with the specified parameters
srun --account=bcnx-delta-cpu --partition=cpu --time=04:00:00 --nodes=1 --ntasks-per-node=1 --mem=32g jupyter-notebook --no-browser --port=$MYPORT --ip=0.0.0.0