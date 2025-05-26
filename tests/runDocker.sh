#!/bin/bash
# docker build --tag 'pytest' .
mv grokking grok
docker build --tag 'pytest' --progress=plain . 2>&1 | tee docker.build.log
mv grok grokking
docker run --init --gpus=all --ipc=host -it 'pytest'
