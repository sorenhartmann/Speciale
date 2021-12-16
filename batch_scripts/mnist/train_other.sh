#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J train_mnist_other
#BSUB -n 4
#BSUB -W 10:00
#BSUB -B
#BSUB -N
#BSUB -R span[hosts=1]
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.6 

cd ~/Documents/Speciale
source .venv/bin/activate

python scripts/inference.py -m \
    +experiment=mnist \
    experiment/mnist=sgd_dropout,sgd_map,vi \
    trainer=best_checkpoint \
    trainer.max_epochs=1000 \
    trainer.max_time="00:06:00:00" \
    ++data.num_workers=3 \
    test=true \
    test_ckpt_path=best \
    ++trainer.progress_bar_refresh_rate=0 \
    ++trainer.gpus=1 