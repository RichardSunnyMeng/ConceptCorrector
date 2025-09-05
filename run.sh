time=$(date "+%Y%m%d-%H%M%S")
NAME=${0%\.*}

ROOT=./
export PYTHONPATH=${ROOT}:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

python generate_diffusers.py