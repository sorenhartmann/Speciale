#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_cifar_small_other
#BSUB -n 4
#BSUB -W 16:00
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
    +experiment=cifar10_small \
    experiment/cifar10_small=sgd_dropout,sgd_map,vi \
    ++trainer.max_epochs=1000 \
    ++trainer.max_time="00:08:00:00" \
    trainer=best_checkpoint_top_ten \
    ++data.num_workers=3 \
    test=true \
    test_ckpt_path=best \
    ++trainer.progress_bar_refresh_rate=0 \
    +extra_callbacks=calculate_calibration \
    ++trainer.gpus=1 