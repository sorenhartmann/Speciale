#!/bin/sh
#BSUB -q hpc
#BSUB -J sghmc_gradients
#BSUB -n 12
#BSUB -W 8:00
#BSUB -B
#BSUB -N
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=1GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.6 

cd ~/Documents/Speciale
source .venv/bin/activate

python src/experiments/sghmc_gradients.py -m hydra/launcher=joblib \
    experiment/sghmc_gradients/estimator@estimator="glob(*)" \
    inference.sampler.variance_estimator.use_estimate="True,False" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_epochs=800