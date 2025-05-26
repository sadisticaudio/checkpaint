#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1

mkdir -p build$CFGTYPE
cd build$CFGTYPE
cmake -DCMAKE_BUILD_TYPE=$CFGTYPE ..
cmake --build . --config $CFGTYPE --target cpptest
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'/media/frye/sda5/boost_1_85_0/stage/lib:/media/frye/sda5/anaconda3/envs/pytorch_env/lib/python3.10/site-packages/torch/lib'
./cpptest
cd ..