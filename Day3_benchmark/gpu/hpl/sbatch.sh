#!/bin/bash
#SBATCH --job-name=hpl   
#SBATCH --nodes=8              
#SBATCH --ntasks-per-node=8    
#SBATCH --cpus-per-task=4      
#SBATCH --gres=gpu:8            
#SBATCH --time=00:15:00          
#SBATCH --account="ACD113087"   
#SBATCH --partition=gtest


export MODULEPATH=$MODULEPATH:/work/HPC_SCAMP/opt/modulefiles/old/twnia2
export MODULEPATH=$MODULEPATH:/work/HPC_SCAMP/opt/modulefiles/twnia2
module purge
module load openmpi/5.0.4
module load cuda/11.4



module list

echo "-------------------------------------------------------------------"
echo " Start at `date`"
echo "-------------------------------------------------------------------"


export LD_LIBRARY_PATH=/opt/ohpc/twcc/intel/2020/update1/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export UCX_MAX_RNDV_RAILS=1
export UCX_TLS=all


mpirun -np 2	\
	-x PATH -x LD_LIBRARY_PATH \
	./xhpl	\
	 HPL.dat >> HPL_result.txt 


echo "-------------------------------------------------------------------"
echo " End at `date`"
echo "-------------------------------------------------------------------"
