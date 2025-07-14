#!/bin/bash
#SBATCH -A ACD113087   # your account
#SBATCH -J HelloWorld   # job's name
#SBATCH -p ct56    # specify partition/queue 
#SBATCH -N 1    # total number of nodes requested         
#SBATCH -n 4    # total number of mpi tasks    
#SBATCH -o hello_world_output.log
#SBATCH -e hello_world_error.log
   
module load gcc/13.2.0
module load openmpi/4.1.6

srun ./hello_world



