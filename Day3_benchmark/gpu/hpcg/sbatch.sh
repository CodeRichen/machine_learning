#!/bin/bash
#SBATCH --job-name=hpcg  
#SBATCH --nodes=1               
#SBATCH --ntasks-per-node=2      
#SBATCH --cpus-per-task=4      
#SBATCH --gres=gpu:2            
#SBATCH --time=00:15:00          
#SBATCH --account="ACD113087"   
#SBATCH --partition=gtest

export MODULEPATH=$MODULEPATH:/work/HPC_SCAMP/opt/modulefiles/old/twnia2
export MODULEPATH=$MODULEPATH:/work/HPC_SCAMP/opt/modulefiles/twnia2
module purge
module load openmpi/5.0.4
module load cuda/11.4


echo "-------------------------------------------------------------------"
echo " Start at `date`"
echo "-------------------------------------------------------------------"

export NCCL_DEBUG=INFO
export UCX_MAX_RNDV_RAILS=1
export UCX_TLS=all

mpirun -np 2	\
	-x PATH -x LD_LIBRARY_PATH \
	./xhpcg	\
	--dat HPCG.dat 


echo "-------------------------------------------------------------------"
echo " End at `date`"
echo "-------------------------------------------------------------------"
