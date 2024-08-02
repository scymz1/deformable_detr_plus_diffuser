#!/bin/bash
#SBATCH --job-name=minghao_geneformer # Job name
#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=minghao.zhou@ufl.edu     # Where to send mail  
#SBATCH --nodes=1                     # Run all processes on a single node  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=250gb                    # Job memory request
#SBATCH --time=140:00:00              # Time limit hrs:min:sec
#SBATCH --output=output.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

module load cuda/12.4.1 intel/2020.0.166 

module load gcc

python diffusion_xl.py