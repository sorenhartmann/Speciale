#!/bin/sh
#BSUB -q hpc
#BSUB -J train_mnist_mcmc
#BSUB -n 2
#BSUB -W 11:00
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

python scripts/inference.py -m hydra/launcher=joblib \
    +experiment=mnist \
    experiment/mnist="glob(sghmc*)" \
    ++trainer.max_epochs=1000 \
    ++data.num_workers=0 \
    ++trainer.progress_bar_refresh_rate=0 \
    trainer.max_time="00:10:00:00" \
    +extra_callbacks=log_temp_and_calculate_calibration \
    test=true
