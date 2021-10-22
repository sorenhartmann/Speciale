#!/bin/sh
#BSUB -q hpc
#BSUB -J cifar
#BSUB -n 24
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

python src/experiments/cifar.py -m hydra/launcher=joblib\
    inference.lr="0.01,0.05,0.1,0.5" \
    ++inference.initial_rho="-2,-3,-4" \
    trainer.gradient_clip_val="0.5,1.,5.,10." \
    trainer.gradient_clip_algorithm="norm" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_epochs=1000