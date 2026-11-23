#!/bin/bash
#SBATCH --job-name=lsc5_predict
#SBATCH --partition=estudiantes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out

source ~/venv_entrenamiento/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd ~/lsc5_ensamble

python -u predecir_test.py
