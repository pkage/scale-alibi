#!/bin/bash
# aiai valluvar setup

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:4
#SBATCH --output=/home/s1734411/scale-alibi/stdout.txt
#SBATCH --error=/home/s1734411/scale-alibi/stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=scale-alibi
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/scale-alibi/

export PROJ_DIR=/home/s1734411/scale-alibi/
export LSCRATCH=/disk/scratch/s1734411

cd $PROJ_DIR

# load python, ensure the env is installed
export PATH=/home/s1734411/py3.11.4/bin:$PATH
cd $PROJ_DIR
poetry install

# check that we've got our vit model checkpoint....
# if ! [ -f ./IN1K-vit.h.14-300e.pth.tar ]; then
#     echo "vit not found, downloading..."
#     curl -L -O -J https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
# else
#     echo "vit has been downloaded already"
# fi

# perf tuning
export OMP_NUM_THREADS=16

# secrets -- wandb setup
source .env

# kickoff
echo "beginning training..."
poetry run wandb offline
poetry run torchrun \
    --nproc_per_node 1 \
    main_ddp.py \
    --vit-ckpt ./IN1K-vit.h.14-300e.pth.tar \
    --batch_size 32 \
    --test_batch_size 32 \
    --epochs 200 \
    --learning_rate 0.001 # 1e-3
