#!/bin/bash

# aiai valluvar copy to storage lock

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --output=/home/s1734411/scale-alibi/stdout.txt
#SBATCH --error=/home/s1734411/scale-alibi/stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=scale-alibi-download
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/scale-alibi/

export PROJ_DIR=/home/s1734411/scale-alibi/
export LSCRATCH=/disk/scratch/s1734411/salibi/datasets

mkdir -p $LSCRATCH

cd $PROJ_DIR

# load python, ensure the env is installed
export PATH=/home/s1734411/py3.11.4/bin:$PATH
cd $PROJ_DIR

# download

mkdir $LSCRATCH/composite

echo "copying sar"
cp ./data/composite/sar.pmtile $LSCRATCH/composite/sar.pmtile

echo "copying visual"
cp ./data/composite/visual.pmtile $LSCRATCH/composite/visual.pmtile

echo "copying hires"
cp ./data/composite/hires_visual.pmtile $LSCRATCH/composite/hires_visual.pmtile

echo "done!"
