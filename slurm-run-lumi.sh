#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p small-g
#SBATCH -t 02:00:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOBID.out logs/latest.out
ln -s $SLURM_JOBID.err logs/latest.err

module load cray-python
source venv/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "START $SLURM_JOBID: $(date)"

srun "$@"

echo "END $SLURM_JOBID: $(date)"
