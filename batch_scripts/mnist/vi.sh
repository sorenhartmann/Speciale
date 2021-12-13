#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J mnist_vi
#BSUB -n 4
#BSUB -W 2:30
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
    +experiment=mnist \
    experiment/mnist=vi \
    ++trainer.max_epochs=1000 \
    ++data.num_workers=3 \
    ++trainer.gpus=1