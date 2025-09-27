#!/bin/bash
for i in $(seq $@)
do
 sbatch run_onr_run.sh $i
done
