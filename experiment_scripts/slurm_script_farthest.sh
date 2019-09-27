#!/bin/bash
#
#SBATCH --job-name=accuracy_comparison_farthest_insertion
#SBATCH --output=/tmp-network/user/ppansari/logs/baseline_farthest_insertion_SLURM_%j-output.log
#SBATCH --error=/tmp-network/user/ppansari/logs/baseline_farthest_insertion_SLURM_%j-errors.err
#
#SBATCH -n 1
#SBATCH -p cpu 
#SBATCH --mem=32g
#SBATCH --cpus-per-task=32
#

source /home/pkpansar/.bashrc
conda activate attention_tsp 

for ((i = 100 ; i <= 1000 ; i = i + 100)); do
echo "Farthest insert - $i"
srun python -m problems.tsp.tsp_baseline farthest_insertion data/tsp/tsp"$i"_test_seed2345.pkl -f 
done
