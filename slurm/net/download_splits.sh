#! /bin/bash

cd ~/scale-alibi

download_single() {
    if ! [ -s data/$1 ]; then
        curl -L https://se7a6ueojehnth4fhglodzchw40rfbuv.lambda-url.us-east-2.on.aws/$1 -o data/$1
    else
        echo "$1 already downloaded"
    fi
}

download_single composite/split_train.npy
download_single composite/split_test.npy
download_single composite/split_val.npy

sbatch slurm/aiai_lock_datasets.sbatch



