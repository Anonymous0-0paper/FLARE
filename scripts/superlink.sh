#!/bin/bash

# output slurm file
OUTPUT_FILE="byebye-badclients-superlink.slurm"

# overwrite file
echo "#!/bin/bash" > "$OUTPUT_FILE"

# sbatch flags
echo "#SBATCH --job-name=superlink" >> "$OUTPUT_FILE"
echo "#SBATCH --output=superlink.out" >> "$OUTPUT_FILE"
echo "#SBATCH --ntasks=1" >> "$OUTPUT_FILE"
echo "#SBATCH --nodes=1" >> "$OUTPUT_FILE"
echo "#SBATCH --partition=IFIall" >> "$OUTPUT_FILE"
echo "#SBATCH --cpus-per-task=2" >> "$OUTPUT_FILE"
echo "#SBATCH --time=12:00:00" >> "$OUTPUT_FILE"

# command for task
echo "srun hostname -i" >> "$OUTPUT_FILE"
echo "srun singularity run --bind /scratch/leon.kiss/byebye-badclients/results:/home/leon.kiss/results \
	byebye-badclients-superlink_latest.sif \
	--insecure" >> "$OUTPUT_FILE"

# submit slurm script
sbatch "$OUTPUT_FILE"
