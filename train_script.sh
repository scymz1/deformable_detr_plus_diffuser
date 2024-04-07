#!/bin/bash
#SBATCH --job-name=minghao_geneformer # Job name
#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=minghao.zhou@ufl.edu     # Where to send mail  
#SBATCH --nodes=1                     # Run all processes on a single node  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=16  
#SBATCH --mem=120gb                    # Job memory request
#SBATCH --time=140:00:00              # Time limit hrs:min:sec
#SBATCH --output=train_logs_continue4.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4

module load cuda/12.2.2 intel/2020.0.166 

module load gcc

GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_single_scale_plus_cross_attn_with_clip.sh --batch_size 18 --resume /blue/yonghui.wu/minghao.zhou/my/Deformable-DETR/exps/r50_deformable_detr_single_scale_plus_cross_attn_with_clip_plus_diffusers/checkpoint0008.pth
