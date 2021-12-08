#!/bin/sh
#BSUB -q hpc
#BSUB -J cifar10-small
#BSUB -n 8
#BSUB -W 24:00
#BSUB -B
#BSUB -N
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=1GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.6 

cd ~/Documents/Speciale
source .venv/bin/activate

python scripts/inference.py -m hydra/launcher=joblib \
    +experiment=cifar10_small \
    experiment/cifar10_small="glob(*)" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_time="00:23:55:00"