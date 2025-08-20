#!/bin/bash

OUTPUT_FILE="byebye-badclients-serverapp.slurm"


echo "#!/bin/bash" > "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"

echo "#SBATCH --job-name=byebye-badclients-serverapp" >> "$OUTPUT_FILE"
echo "#SBATCH --output=byebye-badclients.out" >> "$OUTPUT_FILE"
echo "#SBATCH --partition=IFIgpu" >> "$OUTPUT_FILE"
echo "#SBATCH --ntasks=1" >> "$OUTPUT_FILE"
echo "#SBATCH --gpus-per-task=1" >> "$OUTPUT_FILE"
echo "#SBATCH --cpus-per-task=2" >> "$OUTPUT_FILE"
echo "#SBATCH --time=12:00:00" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"

echo "srun singularity exec --nv \
	--pwd /app \
	--bind /scratch/leon.kiss/byebye-badclients/pyproject.toml:/app/pyproject.toml \
	byebye-badclients-serverapp_mcd.sif \
	flwr run . remote-federation --stream" \
	>> "$OUTPUT_FILE"


sbatch "$OUTPUT_FILE"
