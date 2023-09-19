#!/usr/bin/bash

proj_dir=/home/seang/Dev/Git/CbmSim/
inputs_dir=${proj_dir}data/inputs/
outputs_dir=${proj_dir}data/outputs/
reset_isi_500_dir=${outputs_dir}reset_isi_500_bugaloo/

## awk modifies text streams
## 1) grab all files you want to loop over
## 2) parse the file basename wrt a delimiter ('_')
## 3) select the last element ($3)
## 4) append this element to list of num_resets

cd $reset_isi_500_dir
declare -a reset_nums
reset_nums=$(ls *.pfpcw | awk -F_ '{print $4}' | awk -F. {'print $1'})
cd ${proj_dir}build/

for num in ${reset_nums[@]}; do
	./cbm_sim_gpu_2 --pfpc-off --mfnc-off \
	-i acq_ISI_500.sim \
	-s probe_reset_ISI_500_bugaloo.sess \
	-a reset_isi_500_${num}.pfpcw \
	-r PC \
	-o probe_isi_500_reset_${num}_bugaloo
done

cd ${proj_dir}scripts/
