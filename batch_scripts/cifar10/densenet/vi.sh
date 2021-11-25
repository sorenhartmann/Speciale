#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J cifar10_densenet_vi
#BSUB -n 4
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.6 
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

cd ~/Documents/Speciale
source .venv/bin/activate

python scripts/inference.py \
    +experiment=cifar10_densenet \
    experiment/cifar10_densenet=vi \
    ++trainer.max_epochs=800 \
    ++data.num_workers=2 \
    ++trainer.gpus=1 \
    ++trainer.progress_bar_refresh_rate=0