#!/bin/bash
#
#SBATCH --job-name=accuracy_comparison_am_20
#SBATCH --output=/tmp-network/user/ppansari/logs/baseline_am_20_SLURM_%j-output.log
#SBATCH --error=/tmp-network/user/ppansari/logs/baseline_am_20_SLURM_%j-errors.err
#
#SBATCH -n 1
#SBATCH -p debug 
#SBATCH --mem=32g
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#

source /home/pkpansar/.bashrc
conda activate attention_tsp 

for ((i = 100 ; i <= 1000 ; i = i + 100)); do
echo "attention model trained on 20 nodes graphs - $i"
srun python eval.py data/tsp/tsp"$i"_test_seed2345.pkl --model pretrained/tsp_20 --decode_strategy greedy -f
done
