#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J mnist_sghmc
#BSUB -n 4
#BSUB -W 3:00
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
    experiment/mnist=sghmc \
    ++sampler.lr="5.e-07,1.e-06,2.e-06,5.e-06" \
    ++sampler.alpha="0.01,0.1" \
    ++model.activation_func._target_="torch.nn.ReLU,torch.nn.Sigmoid" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_epochs=1000 \
    ++trainer.gpus=1