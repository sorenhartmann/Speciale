#!/bin/sh
#BSUB -q hpc
#BSUB -J simulated
#BSUB -n 8
#BSUB -W 12:00
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
    +experiment=simulated \
    experiment/simulated="glob(*)" \
    ++trainer.progress_bar_refresh_rate=0