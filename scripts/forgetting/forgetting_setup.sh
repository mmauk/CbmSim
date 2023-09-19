#!/usr/bin/bash

# globals
root_dir="/home/seang/Dev/Git/CbmSim/" # you can always make this an env var
build_dir="${root_dir}build/"
scripts_dir="${root_dir}scripts/"
#declare -a isis=( 200 500 1000 1500 2000 )
declare -a reset_test_nums=( 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 )
declare -a isis=( 500 )

# set -x

# $1 == executable name: string
# $2 == label (not used): string
# $3 == config: associative array
run_trials_on() {
  local exec=$1
  local -n config=$3
	  for isi in ${isis[@]}; do
		echo ./$exec --mfnc-off \
      ${config["input"]}_ISI_${isi}.sim \
      ${config["session"]}_ISI_${isi}.sess \
      ${config["mask"]} \
      ${config["weights"]} \
      ${config["psth"]} \
      ${config["raster"]} \
      ${config["output"]}_ISI_${isi}
	done
}

# $1 == input_base
order_reset_insert_weights() {
  local input_base=$1
  local num_to_reset=$2
  cd $scripts_dir
  . .venv/bin/activate
  for reset_num in ${reset_test_nums[@]}; do
	  for isi in ${isis[@]}; do
      ./order_reset_insert_weights.py \
        $reset_num \
        ${input_base}_ISI_${isi}.pfpc \
        ${input_base}_ISI_${isi}.sim
    done
  done
  deactivate
  cd $build_dir
}

declare -A eq_config=( \
  ["input"]="-i forgetting_build" \
  ["output"]="-o eq" \
  ["session"]="-s eq" \
  ["weights"]="-w PFPC" \
  ["psth"]="-p GR"\
  ["raster"]="" \
)

declare -A acq_config=( \
  ["input"]="-i eq" \
  ["output"]="-o acq_no_probe" \
  ["session"]="-s acq_no_probe" \
  ["weights"]="-w PFPC" \
  ["psth"]="" \
  ["raster"]="-r PC,NC" \
)

declare -A forget_est_config=( \
  ["input"]="-i acq" \
  ["output"]="-o forget_est_20k" \
  ["session"]="-s forget_est" \
  ["weights"]="-w PFPC" \
  ["psth"]="" \
  ["raster"]="-r PC,NC" \
)

declare -A forget_data_collect_config=( \
  ["input"]="-i forget_est" \
  ["output"]="-o forget_data_collect" \
  ["session"]="-s forget_data_collect" \
  ["weights"]="-w PFPC" \
  ["psth"]="" \
  ["raster"]="-r PC,NC" \
)

declare -A acq_nec_suf_config=( \
  ["input"]="" \
  ["output"]="-o acq_nec_suf" \
  ["session"]="-s acq_nec_suf" \
  ["weights"]="" \
  ["psth"]="" \
  ["raster"]="-r PC,NC" \
)

declare -A forget_nec_suf_config=( \
  ["input"]="-i acq" \
  ["output"]="-o forget_nec_suf" \
  ["session"]="-s forget_nec_suf" \
  ["mask"]="-m freeze_isi_500_15000.mask" \
  ["weights"]="" \
  ["psth"]="" \
  ["raster"]="-r PC,NC" \
)

declare -A random_walk_config=( \
  ["input"]="" \
  ["output"]="-o random_walk_cr_test" \
  ["session"]="-s random_walk_cr_test" \
  ["weights"]="" \
  ["psth"]="" \
  ["raster"]="-r PC,NC" \
)

# set +x

