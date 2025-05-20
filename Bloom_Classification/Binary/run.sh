#!/bin/bash
#SBATCH -o slurm.out
#SBATCH -e slurm.err
#SBATCH -p carlsonlab-gpu
#SBATCH -c 4 # CPUs, adjust as needed
#SBATCH --mem-per-cpu=20G # adjust as needed
#SBATCH --gres=gpu:1
python binary_planet.py