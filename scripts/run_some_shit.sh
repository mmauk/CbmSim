#!/usr/bin/bash

isis=(1500 1000)

cd ../build
for isi in ${isis[@]}; do
	out_base_build="isi_${isi}_strp_crs_chk_build"

	out_base_run="isi_${isi}_strp_crs_chk_run"
	curr_date=$(date +%m%d%Y)

	./cbm_sim -b build_file_template_tune_09262022.bld \
			  -o $out_base_build

	./cbm_sim --mfnc-off -i "${out_base_build}_${curr_date}".sim \
						 -s isi_${isi}_acq.sess \
						 -o "$out_base_run" \
						 -p GR \
						 -r PC,BC,SC,IO,NC \
						 &> "$out_base_run".log
done

cd ../scripts

