#!/bin/bash
#SBATCH --job-name=nnlda_100
#SBATCH --time=0-03:00:00
#SBATCH --account=rrg-ubcxzh
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1              # Number of GPUs (per node)
#SBATCH --mem=127000M               # memory (per node)
#SBATCH --mail-type=ALL
#SBATCH --array=0-8:2
#SBATCH --output=/scratch/kbai/results/web_ensembleLDA/dataset1/1/nonof/logs/slurm-%A_%a.out

for ((i=0; i<${SLURM_ARRAY_TASK_STEP};i++))
do
  ./val_run_query.sh $1 $((${SLURM_ARRAY_TASK_ID}+i)) $2 $3 $4 $5 $6 $7 $8 &
done
wait
