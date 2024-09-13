#!/bin/bash
# aiai valluvar upload datasets

#SBATCH --partition=AIAI_GPU
#SBATCH --ntasks=1
#SBATCH --mem=10G
#SBATCH --output=/home/s1734411/scale-alibi/logs/upload-stdout.txt
#SBATCH --error=/home/s1734411/scale-alibi/logs/upload-stderr.txt
#SBATCH --time=12:00:00
#SBATCH --job-name=scale-alibi-upload
#SBATCH --mail-user=p.kage@ed.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/s1734411/scale-alibi/

export PROJ_DIR=/home/s1734411/scale-alibi/
export LSCRATCH=/disk/scratch/s1734411/salibi/checkpoints

mkdir -p $LSCRATCH

cd $PROJ_DIR

PATTERN="*.pth"

# Base URL for upload
BASE_URL="https://hpie463z6jkcbdddih7sa4liwa0nuvvo.lambda-url.us-east-2.on.aws"

# Find and upload files
find "$LSCRATCH" -type f -name "$PATTERN" | while read -r FILE_PATH; do
    # Construct the URL using the file path relative to the base directory
    RELATIVE_PATH="${FILE_PATH#$LSCRATCH/}"
    UPLOAD_URL="$BASE_URL/checkpoints/$RELATIVE_PATH"
    
    # Execute the curl command to upload the file
    echo "Presigning $FILE_PATH to $UPLOAD_URL"
    PRESIGNED_URL=${curl -s $UPLOAD_URL}

    echo "Uploading $FILE_PATH to $RELATIVE_PATH"
    curl -X PUT -L "$PRESIGNED" --data-binary "@$FILE_PATH"
done
