#!/bin/bash
#SBATCH --job-name=minghao_geneformer # Job name
#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=minghao.zhou@ufl.edu     # Where to send mail  
#SBATCH --nodes=1                     # Run all processes on a single node  
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=12
#SBATCH --mem=150gb                    # Job memory request
#SBATCH --time=140:00:00              # Time limit hrs:min:sec
#SBATCH --output=train_logs/defor_detr_plus_clip.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:4

module load cuda/12.4.1 intel/2020.0.166 

module load gcc

# GPUS_PER_NODE=1 ./tools/run_dist_launch.sh 1 ./configs/r50_deformable_detr_single_scale_plus_cross_attn_with_clip.sh --batch_size 18 --resume /blue/yonghui.wu/minghao.zhou/my/deformable_detr_plus_diffuser/exps/r50_deformable_detr_single_scale_plus_cross_attn_with_clip_plus_diffusers/checkpoint0024.pth --epochs 27

#default example: GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh

# GPUS_PER_NODE=6 ./tools/run_dist_launch.sh 6 ./configs/r50_deformable_detr_diffuser_lx.sh --batch_size 40 --epochs 45 --resume /blue/yonghui.wu/minghao.zhou/my/deformable_detr_plus_diffuser/exps/r50_deformable_detr_diffuser_lx/checkpoint0019.pth


GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr_single_scale_plus_cross_attn_with_clip.sh --batch_size 40  --epochs 45