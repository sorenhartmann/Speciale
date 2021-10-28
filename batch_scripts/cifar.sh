#!/bin/sh
#BSUB -q hpc
#BSUB -J cifar
#BSUB -n 20
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

python scripts/inference.py -m hydra/launcher=joblib +experiment=cifar experiment/cifar=vi\
    inference.lr="1e-3,1e-4,1e-5" \
    ++inference.initial_rho=-3 \
    ++trainer.gradient_clip_val="0.1,1.,10." \
    ++trainer.gradient_clip_algorithm="value,norm" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_epochs=1000