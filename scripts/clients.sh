#!/bin/bash

IP="$1"       # First argument: superlink address
BASE_PORT=9094

sbatch clients.slurm "$IP"

