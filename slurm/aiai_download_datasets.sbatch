#! /bin/bash

#!/bin/bash
# aiai valluvar download

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

export OPML_BASE_URL=https://mlops.aws.kage.dev
export OPML_KEY=sk_qGKuKy88p3
export OPML_PROJECT=project

# download

download_dataset() {
    if ! [ -s $LSCRATCH/$1/$2 ]; then
        curl -L "$OPML_BASE_URL/storage/$OPML_PROJECT/download?c=dataset&k=$OPML_KEY&f=$1/$2" -o $LSCRATCH/pr/
    else
        echo "$1/$2 has already been downloaded"
    fi
}

download_dataset composite sar.pmtile
download_dataset composite visual.pmtile
download_dataset composite hires_visual.pmtile
