#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J sghmc_adam
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
    experiment/mnist=sghmc_var_est \
    variance_estimator="interbatch" \
    hydra/sweeper=hp_search \
    hydra.sweeper.study_name="mnist-sghmc-var-est" \
    ++sampler.lr="tag(log,interval(1e-07,1e-05))" \
    ++sampler.alpha="tag(log,interval(0.01,0.1))" \
    ++sampler.estimation_margin="choice(1.3,10,100)" \
    ++model.activation_func._target_="choice(torch.nn.ReLU,torch.nn.Sigmoid)" \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.max_epochs=1600 \
    ++trainer.gpus=1