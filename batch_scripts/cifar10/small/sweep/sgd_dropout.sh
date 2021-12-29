#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J cifar_small_sgd_dropout
#BSUB -n 4
#BSUB -W 12:00
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

python scripts/sweep.py \
    +experiment=cifar10_small \
    experiment/cifar10_small=sgd_dropout \
    sweep/search_space=sgd_dropout \
    sweep.study_name="cifar-small-sgd-dropout" \
    +extra_callbacks=early_stopping \
    ++data.num_workers=3 \
    ++trainer.gpus=1