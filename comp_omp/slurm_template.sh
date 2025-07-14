#!/usr/bin/bash

#SBATCH -A ACD113087
#SBATCH -p ctest
#SBATCH -n 1
#SBATCH -c 14
#SBATCH -J n-body

time ./n-body testcases/X.in X.out
