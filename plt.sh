#!/bin/bash
#SBATCH -p debug
#SBATCH -x bhd0005,bhc0024,bhd0020
#SBATCH -o qjob.log
#SBATCH --mem-per-cpu=4GB
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1

python plotting_rho.py