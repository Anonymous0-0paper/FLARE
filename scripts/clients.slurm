#!/bin/bash


#SBATCH --job-name=clients
#SBATCH --output=clients.out
#SBATCH --partition=IFIgpu
#SBATCH --ntasks=100
#SBATCH --tasks-per-node=8
#SBATCH --time=12:00:00

IP="$1"
BASE_PORT=9094

if [ -z "$IP" ]; then
	echo "Usage: $0 <superlink-ip>"
	exit 1
fi

# Export variables so they're available in srun
export IP BASE_PORT

export TORCHVISION_DATA_DIR=/scratch/flwr_cache/

srun bash -c 'PORT=$((BASE_PORT + SLURM_PROCID))
	echo "Task $SLURM_PROCID starting on $(hostname) with port $PORT"
	singularity run \
	--nv \
	--bind <HOSTPATH>:/scratch \
	byebye-badclients-supernode_latest.sif \
	--insecure \
	--superlink $IP:9092 \
	--clientappio-api-address 0.0.0.0:$PORT \
	--node-config "partition-id=$SLURM_PROCID num-partitions=100"'
