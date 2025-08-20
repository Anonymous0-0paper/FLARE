#!/bin/bash

IP="$1"       # First argument: superlink address
NTASKS="$2"   # Second argument: number of supernodes
BASE_PORT=9094

if [ -z "$IP" ] || [ -z "$NTASKS" ]; then
    echo "Usage: $0 <superlink-ip> <number-of-supernodes>"
    exit 1
fi

OUTPUT_FILE="byebye-badclients.slurm"

# Add commands for each task
CMD=$(sed
	-e "s/{{IP}}/$IP/" \
        -e "s/{{NTASKS}}/$NTASKS/" \
        byebye-badclients-supernode-template.txt)

# Write the output file
echo "$CMD" > "$OUTPUT_FILE"

# Submit the generated Slurm script
sbatch "$OUTPUT_FILE"
