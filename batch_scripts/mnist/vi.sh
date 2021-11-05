#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J mnist_vi
#BSUB -n 4
#BSUB -W 24:00
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

python scripts/inference.py -m \
    +experiment=mnist \
    experiment/mnist=vi \
    inference.lr="1.e-03,1.e-04,1.e-05" \
    inference.prior_spec.default_prior.log_sigma_1="0,-1,-2" \
    inference.prior_spec.default_prior.log_sigma_2="-6,-7,-8" \
    inference.kl_weighting_scheme._target_="src.inference.vi.ExponentialKLWeight,src.inference.vi.ConstantKLWeight" \
    ++trainer.progress_bar_refresh_rate=0 \
    trainer.max_epochs=800 \
    ++data.num_workers=4 \
    ++trainer.gpus=1