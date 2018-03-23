#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH -n 2
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END

module add openblas/0.2.20
module add cuda/8.0
module add cudnn/5.1-cuda-8.0

export CPATH=$CPATH:~/.local/include
export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

python3 train_vae.py --save --gpu
python3 train_discriminator.py --save --gpu
python3 test.py --model ctextgen --gpu
