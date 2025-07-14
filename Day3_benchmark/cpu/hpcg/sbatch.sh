#!/bin/bash
#SBATCH -A ACD113087         # Account name/project number
#SBATCH -J hpcg          # Job name
#SBATCH -p ct56         # Partition name
#SBATCH -n 20              # Number of MPI tasks (i.e. processes)
#SBATCH -c 1            # Number of cores per MPI task
#SBATCH -t 10:00        # Wall time limit (days-hrs:min:sec)

export MODULEPATH=$MODULEPATH:/work/HPC_SCAMP/opt/modulefiles/twnia2
module purge
module load openmpi/4.1.5


# {TODO1} go to hpcg directory
mpirun -np 20 --bind-to none ./xhpcg 



