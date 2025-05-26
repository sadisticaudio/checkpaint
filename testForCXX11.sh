#!/bin/bash

testlib=$1

tempStrings=$(strings $testlib | grep cxx11)
if (( ${#tempStrings[@]} == 0 )); then
    echo "No cxx11 abi strings found in binary"
fi
if (( ${#tempStrings[@]} != 0 )); then
    echo "cxx11 abi strings found in binary"
    printf '%s\n' "${tempStrings[@]}"
fi
