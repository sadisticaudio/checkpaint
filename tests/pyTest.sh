#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1
./build.sh $CFGTYPE
cp ./cimporter.py ./build$CFGTYPE/
eval "$(conda shell.bash hook)"
# conda activate pytorch_env
conda activate jupyter_env
cd ./build$CFGTYPE
python cimporter.py
# conda deactivate
# cd ../
