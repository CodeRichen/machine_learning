#!/usr/bin/bash
#SBATCH -A ACD113087
#SBATCH -p ctest
#SBATCH -n 10
#SBATCH --exclusive
#SBATCH -J pi_calc
#SBATCH -e %j.e
#SBATCH -o %j.out

module load gcc/12.3.0 openmpi/4.1.6

# mpirun -n <# of process> ./pi_calc <# of tests>
time mpirun -n 10 ./pi_calc 10000
