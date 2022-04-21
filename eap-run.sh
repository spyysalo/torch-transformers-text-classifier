#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH -p eap
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:mi100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000069
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOBID.out logs/latest.out
ln -s $SLURM_JOBID.err logs/latest.err

source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun "$@"

seff $SLURM_JOBID

echo "END $SLURM_JOBID: $(date)"
