#!/usr/bin/bash

# globals
root_dir="/home/seang/Dev/Git/CbmSim/"
build_dir="${root_dir}build/"
scripts_dir="${root_dir}scripts/"
this_isi=2000
bunny_num=1

set -x

#commence the automation
cd $build_dir
for i in {0..7}; do # 7 intervals of 250 between ISI 2000 and ISI 250
	for j in {1..3}; do # 7 simulations per isi
		./cbm_sim -b build_file_template_tune_09262022.bld -o isi_func_bunny_${bunny_num}
		./cbm_sim -i isi_func_bunny_${bunny_num}.sim -s isi_${this_isi}_isi_func.sess \
				  -o isi_${this_isi}_sim_${bunny_num}_isi_func -r PC -p MF
		(( bunny_num++ ))
	done
	(( this_isi -= 250 ))
done
cd $scripts_dir

