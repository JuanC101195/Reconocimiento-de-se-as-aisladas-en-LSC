#!/bin/bash
#SBATCH --job-name=lsc5_ens3
#SBATCH --partition=estudiantes
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out

source ~/venv_entrenamiento/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python ~/lsc5_ensamble/train_lsc5.py \
  --data-root ~/Dataset \
  --train-dir ~/Dataset/train/landmarks \
  --out-dir ~/resultados/lsc5_t10_ens3_cronos \
  --device cpu \
  --threads 4 \
  --train-final
