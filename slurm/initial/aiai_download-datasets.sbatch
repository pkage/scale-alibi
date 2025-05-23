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
poetry install

# check that we've got our vit model checkpoint....
# if ! [ -f ./IN1K-vit.h.14-300e.pth.tar ]; then
#     echo "vit not found, downloading..."
#     curl -L -O -J https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar
# else
#     echo "vit has been downloaded already"
# fi

# small batch artisanal tilesets

if ! [ -f $LSCRATCH/visual_tiles_small.pmtile ]; then
    echo "visual_tiles_small.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/visual_tiles_small.pmtile -o $LSCRATCH/visual_tiles_small.pmtile
else
    echo "visual_tiles_small.pmtile has already been downloaded."
fi

if ! [ -f $LSCRATCH/sar_tiles_small.pmtile ]; then
    echo "sar_tiles_small.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/sar_tiles_small.pmtile -o $LSCRATCH/sar_tiles_small.pmtile
else
    echo "sar_tiles_small.pmtile has already been downloaded."
fi

if ! [ -f $LSCRATCH/hires_visual_tiles_small.pmtile ]; then
    echo "hires_visual_tiles_small.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/hires_visual_tiles_small.pmtile -o $LSCRATCH/hires_visual_tiles_small.pmtile
else
    echo "hires_visual_tiles_small.pmtile has already been downloaded."
fi

# big ones

if ! [ -f $LSCRATCH/visual_tiles.pmtile ]; then
    echo "visual_tiles.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/visual_tiles.pmtile -o $LSCRATCH/visual_tiles.pmtile
else
    echo "visual_tiles.pmtile has already been downloaded."
fi

if ! [ -f $LSCRATCH/sar_tiles.pmtile ]; then
    echo "sar_tiles.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/sar_tiles.pmtile -o $LSCRATCH/sar_tiles.pmtile
else
    echo "sar_tiles.pmtile has already been downloaded."
fi

if ! [ -f $LSCRATCH/hires_visual_tiles.pmtile ]; then
    echo "hires_visual_tiles.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/hires_visual_tiles.pmtile -o $LSCRATCH/hires_visual_tiles.pmtile
else
    echo "hires_visual_tiles.pmtile has already been downloaded."
fi

# micro ones

if ! [ -f $LSCRATCH/visual_tiles_micro.pmtile ]; then
    echo "visual_tiles_micro.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/visual_tiles_micro.pmtile -o $LSCRATCH/visual_tiles_micro.pmtile
else
    echo "visual_tiles_micro.pmtile has already been downloaded."
fi

if ! [ -f $LSCRATCH/sar_tiles_micro.pmtile ]; then
    echo "sar_tiles_micro.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/sar_tiles_micro.pmtile -o $LSCRATCH/sar_tiles_micro.pmtile
else
    echo "sar_tiles_micro.pmtile has already been downloaded."
fi

if ! [ -f $LSCRATCH/hires_visual_tiles_micro.pmtile ]; then
    echo "hires_visual_tiles_micro.pmtile not found, downloading..."
    curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/hires_visual_tiles_micro.pmtile -o $LSCRATCH/hires_visual_tiles_micro.pmtile
else
    echo "hires_visual_tiles_micro.pmtile has already been downloaded."
fi

