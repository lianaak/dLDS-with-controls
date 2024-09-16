#!/bin/bash

# Parameters
#SBATCH --cpus-per-task=4
#SBATCH --error=/lisc/user/akobian/dLDS-with-controls/dLDS_discrete/dlds_discrete/cdlds/submitit_logs/%j_0_log.err
#SBATCH --gpus-per-node=1
#SBATCH --job-name=submitit
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --open-mode=append
#SBATCH --output=/lisc/user/akobian/dLDS-with-controls/dLDS_discrete/dlds_discrete/cdlds/submitit_logs/%j_0_log.out
#SBATCH --signal=USR2@90
#SBATCH --time=60
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /lisc/user/akobian/dLDS-with-controls/dLDS_discrete/dlds_discrete/cdlds/submitit_logs/%j_%t_log.out --error /lisc/user/akobian/dLDS-with-controls/dLDS_discrete/dlds_discrete/cdlds/submitit_logs/%j_%t_log.err /lisc/app/conda/miniconda3/bin/python -u -m submitit.core._submit /lisc/user/akobian/dLDS-with-controls/dLDS_discrete/dlds_discrete/cdlds/submitit_logs
