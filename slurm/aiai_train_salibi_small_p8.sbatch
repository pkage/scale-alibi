#!/bin/bash
# aiai valluvar setup

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:A40:4
#SBATCH --output=/home/s1734411/scale-alibi/logs/salibi_small_stdout.txt
#SBATCH --error=/home/s1734411/scale-alibi/logs/salibi_small_stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=sa-p8-small
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/scale-alibi/

export PROJ_DIR=/home/s1734411/scale-alibi/
export LSCRATCH=/disk/scratch/s1734411/salibi

mkdir -p $LSCRATCH/checkpoints/salibi_small

cd $PROJ_DIR

# load python, ensure the env is installed
export PATH=/home/s1734411/py3.11.4/bin:$PATH
cd $PROJ_DIR
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -e .

# perf tuning
export OMP_NUM_THREADS=16

# secrets -- wandb setup
source .env

# kickoff
echo "beginning training..."
wandb offline

salibi hardware


salibi train \
	--lores $LSCRATCH/datasets/visual_tiles_small.pmtile \
	--radar $LSCRATCH/datasets/sar_tiles_small.pmtile \
	--hires $LSCRATCH/datasets/hires_visual_tiles_small.pmtile \
	--ckpts $LSCRATCH/checkpoints/salibi_small_p8 \
	--run-name salibi_small_p8 \
	--learning-rate 1e-4 \
	--epochs 50 \
	--batch-size 24 \
	--mask-ratio 0.4 \
	--device cuda \
    --patch-size 8 \
    --patch-count 256 \
    --amp \
    --half-resolution


