#!/bin/bash

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1

rm -rf build$CFGTYPE
./build.sh $CFGTYPE
