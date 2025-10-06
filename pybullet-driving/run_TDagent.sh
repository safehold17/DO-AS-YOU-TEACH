#!/bin/bash
#BATCH -A research
#SBATCH --gres=gpu:1 
#SBATCH -c 40
#SBATCH --time=3-00:00:17
module add singularity/3.7.2
module add cuda/10.2
module add cudnn/7.6.5-cuda-10.2


source ~/miniconda3/bin/activate
conda activate selfplay

cd /home/tanmay.kumar/TD3/
python3 main.py
