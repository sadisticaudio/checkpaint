#!/bin/bash

rootDir=${PWD}

mkdir -p /root/build
# rm -rf /root/build/checkpaint*

CFGTYPE=RELEASE
[[ $1 ]] && CFGTYPE=$1

tVers="2.0 2.1 2.2 2.3 2.4"
[[ $2 ]] && tVers=$2
cVers="11.8 12.1 12.4"
[[ $3 ]] && cVers=$3
pVers="3.8 3.9 3.10 3.11 3.12"
[[ $4 ]] && pVers=$4

ver="0.0.1"

echo "torch versions = $tVers"
echo "cuda versions = $cVers"
echo "python versions = $pVers"

tVersLegal=()

for tVer in $tVers
do
    cVersLegal=("11.8")
    if [[ "$tVer" == "2.4" ]]; then
        cVersLegal+=("12.1")
    fi
    if [[ $tVer == "2.5" ]]; then
        cVersLegal+=("12.1")
        cVersLegal+=("12.4")
    fi
    for cVer in $cVers
    do
        if [[ ! " ${cVersLegal[*]} " =~ [[:space:]]${cVer}[[:space:]] ]]; then
            echo "skipping torch-${tVer} cuda-${cVer}"
            continue
        fi
        pVersLegal=("3.8" "3.9" "3.10" "3.11")
        if [[ $tVer == "2.4" ]]; then
            pVersLegal=("3.8" "3.9" "3.10" "3.11" "3.12")
        fi
        if [[ $tVer == "2.5" ]]; then
            pVersLegal=("3.9" "3.10" "3.11" "3.12")
        fi
        for pVer in $pVers
        do
            if [[ ! " ${pVersLegal[*]} " =~ [[:space:]]${pVer}[[:space:]] ]]; then
                echo "skipping torch-${tVer} cuda-${cVer} python-${pVer}"
                continue
            fi
            echo "building torch-${tVer} cuda-${cVer} python-${pVer}"
            pName=${pVer//.}
            export ptVer=$tVer
            export cudaVer=$cVer
            export pyVer=$pVer
            source activateBuildEnv
            newProject="checkpaint_${tVer}_${cVer}-${ver}-${pVer}"
            projectDir="/root/build/${newProject}"
            rm -rf ${projectDir}
            mkdir ${projectDir}
            projectDir=${projectDir}/checkpaint
            mkdir ${projectDir}
            cp -R ${rootDir}/template/* ${projectDir}/
            cd ${projectDir}
            echo "projectDir ${projectDir}"
            sed -i -e "s/CFG_TYPE/${CFGTYPE}/g" pyproject.toml
            sed -i -e "s/CHECKPAINT_VERSION/${ver}/g" pyproject.toml
            sed -i -e "s/TORCH_VERSION/${tVer}/g" pyproject.toml
            sed -i -e "s/CUDA_VERSION/${cVer}/g" pyproject.toml
            sed -i -e "s/PYTHON_VERSION/${pVer}/g" pyproject.toml

            pipx run --python /opt/python/cp${pName}-cp${pName}/bin/python build --outdir /root/build/output ${projectDir}
            # mkdir -p /root/build_artifacts
            # cp -r ${rootDir}/builds/checkpaint-${ver}_${tVer}_${cVer}-${pVer} /root/build_artifacts/
            # cd ${rootDir}/builds/checkpaint-${ver}_${tVer}_${cVer}-${pVer}/dist
            # wheel unpack checkpaint-${ver}-cp${pName}-cp${pName}-linux_x86_64.whl
            # cd checkpaint-${ver}/checkpaint
            # tempStrings=$(strings _core.cpython-${pName}-x86_64-linux-gnu.so | grep cxx11)
            # if (( ${#tempStrings[@]} == 0 )); then
            #     echo "No cxx11 abi strings found in binary"
            # fi
            # if (( ${#tempStrings[@]} != 0 )); then
            #     echo "cxx11 abi strings found in binary"
            #     echo ${tempStrings}
            # fi
        done
    done
done