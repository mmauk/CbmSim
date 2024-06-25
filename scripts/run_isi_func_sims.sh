#!/usr/bin/bash

# globals
root_dir="/home/seang/Dev/Git/CbmSim/"
build_dir="${root_dir}build/"
scripts_dir="${root_dir}scripts/"
this_isi=2000
bunny_num=0
gpu_num=$1
#set -x

#commence the automation
cd $build_dir
for i in {0..3}; do # 7 intervals of 250 between ISI 2000 and ISI 250
	for j in {1..2}; do # 2 simulations per isi
		echo ./cbm_sim_gpu_$gpu_num --verbose -o isi_func_bunny_${bunny_num}
		echo ./cbm_sim_gpu_$gpu_num --verbose -i isi_func_bunny_${bunny_num}.sim -s isi_func_isi_${this_isi}.json \
				  -o isi_func_frac_cs_0_01_coll_off_isi_${this_isi}_sim_${bunny_num} -r PC
		(( bunny_num++ ))
	done
	(( this_isi -= 250 ))
done

cd $scripts_dir

