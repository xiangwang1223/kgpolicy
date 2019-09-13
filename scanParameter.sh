#!/bin/bash

for lr in 0.0001 0.0005 0.001 0.005
do
    for regs in 1e-8 1e-7 1e-6 1e-5 1e-4 1e-3
    do
        echo "---------------------------------------------------------" >> train_fm.log
        echo Curren parameter: lr - $lr, regs - $regs >> train_fm.log
        python Main.py --resume True --lr $lr --regs $regs >> train_fm.log
        echo "---------------------------------------------------------" >> train_fm.log
    done
done
