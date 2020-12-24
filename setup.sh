#!/bin/bash
# ----------------------------------------------------
# 
# This script is to initiate Environment for KGPolicy
# Dependency including 
# - python=3.6
# - torch=1.1.0
# - torch_geometric=1.3.0
# - tqdm=4.32.2
# - sklearn=0.21.2
# -----------------------------------------------------

current_gcc_ver="$(gcc -dumpversion)"
required_gcc_ver="4.9.0"
if [ "$(printf '%s\n' "$required_gcc_ver" "$current_gcc_ver" | sort -V | head -n1)" = "$required_gcc_ver" ]; then 
    echo "GCC version ${current_gcc_ver}"
else
    echo "GCC version less than ${required_gcc_ver}"
    exit 1
fi

MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
echo ${green}[INFO] "System python version: "${green}$(python -c 'import sys; print(sys.version)')
if [ $MAJOR != "3" ] 
then
    echo "${red}[ERROR] We require python==3.6..."
    echo [EXIT]
    exit 1
else
    if [ $MINOR -gt "7" ]
    then
    echo "${red}[ERROR] We do not support python==3.8..."
    echo "${red}        Please use python==3.6..."
    echo [EXIT]
    exit 1
    fi
fi

# torch_geometric needs torch as prerequisite
echo "${green}[INFO] Begin to install torch==1.1.0"
pip install torch==1.1.0

DOCROOT=$(pwd)
echo "${green}[INFO] Current path" $DOCROOT
pip install -r "$DOCROOT/requirements.txt"
echo ${green}[INFO] SETUP FINISHED
