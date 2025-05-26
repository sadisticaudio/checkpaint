#! /bin/bash

pyVers='3.8 3.9 3.10 3.11 3.12'
for pyVer in $pyVers; do
  pyName="${pyVer//./}"
  eval "$(conda shell.bash hook)"
  conda_env="py${pyName}"
  conda activate $conda_env
  wheel_file="checkpaint-0.0.1-cp${pyName}-cp${pyName}-linux_x86_64.whl"
  echo "wheel_file = $wheel_file"
  pip install $wheel_file
  # python cimporter.py
  conda deactivate
done