#!/bin/sh
#
#PBS -N name
#PBS -l walltime=24:00:00

# run python program
/usr/bin/python /home/npoilvert/connectome/compute_tm.py --epoch 3 --mem_limit 2147483648 1024 1 1 4 4
