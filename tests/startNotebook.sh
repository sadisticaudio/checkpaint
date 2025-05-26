#!/bin/bash

# source /media/frye/sda5/anaconda3/bin/activate
# jupyter notebook stop 9999
# # python3 -m notebook --no-browser
# jupyter notebook --no-browser --port=9999

eval "$(conda shell.bash hook)"
conda activate jupyter_env
cd /media/frye/sda5/
jupyter lab  --no-browser --port=9999
