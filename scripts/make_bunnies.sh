#!/usr/bin/bash

declare -a base_name="bunny"
declare -a sim_ext="sim"

declare -a build_file="build_file_tune_09262022.bld"
declare -a sess_file="practice_sess_file_3.sess"

declare -a command="cbm_sim"

declare -a in_bunny
declare -a out_bunny

# set -x # sets all commands to be echoed during script execution

if [[ -z "$1" ]]; then
	printf "[ERROR]: Number of bunnies not specified. You must provide a second\n"
	printf "[ERROR]: argument indicating the number of bunnies to be created.\n"
	printf "[ERROR]: Exiting...\n"
	exit 1
fi

printf "[INFO]: Entering build directory...\n"
cd ../build/

printf "[INFO]: Constructing $1 bunnies...\n\n"
for (( i=1; i<=$1; i++)); do
	printf "[INFO]: Generating bunny number $i...\n"
	./"$command" "$build_file" "${base_name}_${i}"
	printf "[INFO]: Finished generating bunny number $i.\n"
done
printf "\n[INFO]: Finished constructing $1 bunnies.\n"

printf "[INFO]: Exiting build directory...\n"
cd ../scripts
printf "[INFO]: Back in scripts directory. Exiting successfully...\n"

