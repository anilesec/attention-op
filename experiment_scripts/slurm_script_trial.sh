#!/bin/bash
#
#SBATCH --job-name=wkool_train_op_dist_10_rollout
#SBATCH --output=/tmp-network/user/ppansari/logs/attention_train_10_SLURM_%j-output.log
#SBATCH --error=/tmp-network/user/ppansari/logs/attention_train_10_SLURM_%j-errors.err
#
#SBATCH -n 2
#SBATCH -p gpu-multi
#
#SBATCH --gres=gpu:2


source /home/pkpansar/.bashrc
conda activate attention_tsp 
srun python /nfs/team/mlo/ppansari/tour_generation/code/attention-learn-to-route/run.py --problem op --data_distribution dist --graph_size 10 --baseline rollout --load_path /nfs/team/mlo/ppansari/tour_generation/code/attention-learn-to-route/pretrained/op_dist_100/epoch-99.pt --run_name 'op_dist_10_rollout'

cp /tmp-network/user/ppansari/logs/attention_train_10_SLURM_${SLURM_JOB_ID}-output.log
cp /tmp-network/user/ppansari/logs/attention_train_10_SLURM_${SLURM_JOB_ID}-errors.err

#cp /tmp-network/user/tformal/logs/gat-conv-image-rank/train-graph-ranking-SLURM-${SLURM_JOB_ID}.log $CHECKPOINT_DIR
#cp /tmp-network/user/tformal/logs/gat-conv-image-rank/train-graph-ranking-SLURM-${SLURM_JOB_ID}-errors.err $CHECKPOINT_DIR
