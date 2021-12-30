#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_cifar_densenet_vi
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
    +experiment=cifar10_densenet \
    experiment/cifar10_densenet=vi \
    inference.lr=2e-4,5e-4,1e-3 \
    ++trainer.max_epochs=1000 \
    ++trainer.max_time="00:07:00:00" \
    trainer=best_checkpoint \
    ++data.num_workers=3 \
    test=true \
    test_ckpt_path=best \
    ++trainer.progress_bar_refresh_rate=1 \
    +extra_callbacks=calculate_calibration \
    ++trainer.gpus=1 