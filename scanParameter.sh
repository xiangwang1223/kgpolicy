#!/bin/bash

for lr in 0.0001 0.0005 0.001
do
    for regs in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3
    do
        echo "---------------------------------------------------------"
        echo Curren parameter: lr - $lr, regs - $regs
        python Main.py --resume True --rlr $lr --regs $regs
        echo "---------------------------------------------------------"
    done
done
