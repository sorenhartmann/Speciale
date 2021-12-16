#!/bin/sh
#BSUB -q hpc
#BSUB -J train_mnist_mcmc
#BSUB -n 8
#BSUB -W 10:00
#BSUB -B
#BSUB -N
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.6 

cd ~/Documents/Speciale
source .venv/bin/activate

python scripts/inference.py -m hydra/launcher=joblib\
    +experiment=mnist \
    experiment/mnist="glob(sghmc*)" \
    +extra_callbacks=make_sample_curve_and_log_temp \
    ++trainer.max_epochs=1000 \
    ++data.num_workers=0 \
    ++trainer.progress_bar_refresh_rate=0 \
    test=true