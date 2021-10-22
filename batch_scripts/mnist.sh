#!/bin/sh
#BSUB -q hpc
#BSUB -J sghmc_gradients
#BSUB -n 4
#BSUB -W 1:00
#BSUB -B
#BSUB -N
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=1GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.6 

cd ~/Documents/Speciale
source .venv/bin/activate

python src/experiments/mnist.py -m hydra/launcher=joblib \
    experiment/mnist/group@group="glob(*)" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_epochs=800