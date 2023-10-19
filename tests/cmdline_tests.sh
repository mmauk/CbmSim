#!/usr/bin/bash

# globals
root_dir="/home/seang/Dev/Git/CbmSim/"
data_in_dir="${root_dir}data/inputs/"
data_out_dir="${root_dir}data/outputs/"
build_dir="${root_dir}build/"
debug_dir="${build_dir}debug/"
tests_dir="${root_dir}tests/"

build_file="build_file_template.bld"
sess_file="TESTS.sess"
binary="${build_dir}cbm_sim"
declare -A rast_exts=( \
	[mf]=".mfr" [gr]=".grr" \
	[go]=".gor" [pc]=".pcr" \
	[bc]=".bcr" [sc]=".scr" \
	[nc]=".ncr" [io]=".ior" \
)

declare -A psth_exts=( \
	[mf]=".mfp" [gr]=".grp" \
	[go]=".gop" [pc]=".pcp" \
	[bc]=".bcp" [sc]=".scp" \
	[nc]=".ncp" [io]=".iop" \
)

passed_tests=0
num_tests=

if [ $# -eq 0 ]; then
	printf "Running all tests...\n"
	num_tests=50
elif [ $# -eq 1 ]; then
	if [ "$1" == "build" ]; then
		printf "Running only build-mode tests...\n"
		num_tests=10
	elif [ "$1" == "run" ]; then
		printf "Running only run-mode tests...\n"
		num_tests=40
	elif [ "$1" == "connect" ]; then
		printf "Running only connectivity-mode tests...\n"
		num_tests=4
	fi
fi

# workflow 0 test cases: building a simulation
if [[ $# -eq 0 || ( $# -eq 1 && "$1" == "build" ) ]]; then
	## valid test cases
	
	if ! $binary -b $build_file -o TEST_CASE_1
	then
		printf "TEST CASE 1 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_1" ]
		then
			printf "TEST CASE 1 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_1' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_1/TEST_CASE_1.sim" ]
		then
			printf "TEST CASE 1 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_1.sim' was not produced\n"
		else
			printf "TEST CASE 1 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	 
	if ! $binary -b $build_file -o TEST_CASE_2 -r PC,GO -p PC,GO -w PFPC,MFNC
	then
		printf "TEST CASE 2 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_2" ]
		then
			printf "TEST CASE 2 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_2' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_2/TEST_CASE_2.sim" ]
		then
			printf "TEST CASE 2 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_2.sim' was not produced\n"
		else
			printf "TEST CASE 2 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $binary -b $build_file -o TEST_CASE_3 -v TUI -r PC,GO -p PC,GO -w PFPC,MFNC
	then
		printf "TEST CASE 1 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_3" ]
		then
			printf "TEST CASE 3 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_3' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_3/TEST_CASE_3.sim" ]
		then
			printf "TEST CASE 3 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_3.sim' was not produced\n"
		else
			printf "TEST CASE 3 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	## invalid test cases
	
	err="$( { $binary -b $build_file -o TEST_CASE_4 -v GUI -r PC,GO -p PC,GO -w PFPC,MFNC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify visual mode 'GUI' in build mode. Exiting..." ]]
	then
		printf "TEST CASE 4 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 4 FAILED\n"
	fi
	
	err="$( { $binary -b $build_file -o TEST_CASE_5 -s $sess_file -r PC,GO -p PC,GO -w PFPC,MFNC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify both session and build file. Exiting..." ]]
	then
		printf "TEST CASE 5 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 5 FAILED\n"
	fi
	
	err="$( { $binary -b $build_file -s $sess_file -r PC,GO -p PC,GO -w PFPC,MFNC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify both session and build file. Exiting..." ]]
	then
		printf "TEST CASE 6 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 6 FAILED\n"
	fi
	
	err="$( { $binary -b $build_file -r PC,GO -p PC,GO -w PFPC,MFNC > /dev/null; } 2>&1 )"
	if [[ $err =~ "You must specify an output basename. Exiting..." ]]
	then
		printf "TEST CASE 7 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 7 FAILED\n"
	fi
	
	err="$( { $binary -b $build_file -v TUI > /dev/null; } 2>&1 )"
	if [[ $err =~ "You must specify an output basename. Exiting..." ]]
	then
		printf "TEST CASE 8 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 8 FAILED\n"
	fi
	
	err="$( { $binary -b $build_file -v GUI > /dev/null; } 2>&1 )"
	if [[ $err =~ "You must specify an output basename. Exiting..." ]]
	then
		printf "TEST CASE 9 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 9 FAILED\n"
	fi
	
	err="$( { $binary -b $build_file -i TEST_INPUT.sim > /dev/null; } 2>&1 )"
	if [[ $err =~ "You must specify an output basename. Exiting..." ]]
	then
		printf "TEST CASE 10 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 10 FAILED\n"
	fi
fi

if [[ $# -eq 0 || ( $# -eq 1 && "$1" == "run" ) ]]; then
	# workflow 1 test cases: building a simulation
	
	workflow_1_basename="WORKFLOW_1_INPUT"
	workflow_1_input="${workflow_1_basename}.sim"
	if [ -e "${data_out_dir}${workflow_1_basename}/${workflow_1_input}" ]; then
		printf "workflow 1 input sim found. No need to generate a new one...\n"
	else
		printf "Generating simulation for workflow 1 test cases...\n"
		$binary -b $build_file -o $workflow_1_basename
	fi
	
	## valid test cases
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_11 > /dev/null 2>&1 )
	then
		printf "TEST CASE 11 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_11" ]
		then
			printf "TEST CASE 11 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_11' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_11/TEST_CASE_11.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_11/TEST_CASE_11.txt" ] 
		then
			printf "TEST CASE 11 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_11.sim' and info file 'TEST_CASE_11.txt' were not produced\n"
		else
			printf "TEST CASE 11 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_12 -r MF > /dev/null 2>&1 )
	then
		printf "TEST CASE 12 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_12" ]
		then
			printf "TEST CASE 12 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_12' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_12/TEST_CASE_12.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_12/TEST_CASE_12.txt" ] 
		then
			printf "TEST CASE 12 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_12.sim' and info file 'TEST_CASE_12.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_12/TEST_CASE_12${rast_exts[mf]}" ]
		then
			printf "TEST CASE 12 FAILED\n"
			printf "\tREASON: Raster file 'TEST_CASE_12${rast_exts[mf]}' was not produced\n"
		else
			printf "TEST CASE 12 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_13 -r MF,GO > /dev/null 2>&1 )
	then
		printf "TEST CASE 13 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_13" ]
		then
			printf "TEST CASE 13 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_13' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_13/TEST_CASE_13.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_13/TEST_CASE_13.txt" ] 
		then
			printf "TEST CASE 13 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_13.sim' and info file 'TEST_CASE_13.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_13/TEST_CASE_13${rast_exts[mf]}" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_13/TEST_CASE_13${rast_exts[go]}" ]
		then
			printf "TEST CASE 13 FAILED\n"
			printf "\tREASON: Raster files 'TEST_CASE_13${rast_exts[mf]}' and 'TEST_CASE_13${rast_exts[go]}' were not produced\n"
		else
			printf "TEST CASE 13 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_14 -r GO,MF > /dev/null 2>&1 )
	then
		printf "TEST CASE 14 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_14" ]
		then
			printf "TEST CASE 14 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_14' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_14/TEST_CASE_14.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_14/TEST_CASE_14.txt" ] 
		then
			printf "TEST CASE 14 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_14.sim' and info file 'TEST_CASE_14.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_14/TEST_CASE_14${rast_exts[mf]}" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_14/TEST_CASE_14${rast_exts[go]}" ]
		then
			printf "TEST CASE 14 FAILED\n"
			printf "\tREASON: Raster files 'TEST_CASE_14${rast_exts[mf]}' and 'TEST_CASE_14${rast_exts[go]}' were not produced\n"
		else
			printf "TEST CASE 14 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_15 -p MF > /dev/null 2>&1 )
	then
		printf "TEST CASE 15 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_15" ]
		then
			printf "TEST CASE 15 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_15' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_15/TEST_CASE_15.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_15/TEST_CASE_15.txt" ] 
		then
			printf "TEST CASE 15 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_15.sim' and info file 'TEST_CASE_15.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_15/TEST_CASE_15${psth_exts[mf]}" ]
		then
			printf "TEST CASE 15 FAILED\n"
			printf "\tREASON: PSTH file 'TEST_CASE_15${psth_exts[mf]}' was not produced\n"
		else
			printf "TEST CASE 15 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_16 -p MF,GO > /dev/null 2>&1 )
	then
		printf "TEST CASE 16 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_16" ]
		then
			printf "TEST CASE 16 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_16' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_16/TEST_CASE_16.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_16/TEST_CASE_16.txt" ] 
		then
			printf "TEST CASE 16 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_16.sim' and info file 'TEST_CASE_16.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_16/TEST_CASE_16${psth_exts[mf]}" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_16/TEST_CASE_16${psth_exts[go]}" ]
		then
			printf "TEST CASE 16 FAILED\n"
			printf "\tREASON: PSTH files 'TEST_CASE_16${psth_exts[mf]}' and 'TEST_CASE_16${psth_exts[go]}' were not produced\n"
		else
			printf "TEST CASE 16 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_17 -p GO,MF > /dev/null 2>&1 )
	then
		printf "TEST CASE 17 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_17" ]
		then
			printf "TEST CASE 17 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_17' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_17/TEST_CASE_17.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_17/TEST_CASE_17.txt" ] 
		then
			printf "TEST CASE 17 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_17.sim' and info file 'TEST_CASE_17.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_17/TEST_CASE_17${psth_exts[mf]}" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_17/TEST_CASE_17${psth_exts[go]}" ]
		then
			printf "TEST CASE 17 FAILED\n"
			printf "\tREASON: PSTH files 'TEST_CASE_17${psth_exts[mf]}' and 'TEST_CASE_17${psth_exts[go]}' were not produced\n"
		else
			printf "TEST CASE 17 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_18 -w PFPC > /dev/null 2>&1 )
	then
		printf "TEST CASE 18 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_18" ]
		then
			printf "TEST CASE 18 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_18' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_18/TEST_CASE_18.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_18/TEST_CASE_18.txt" ] 
		then
			printf "TEST CASE 18 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_18.sim' and info file 'TEST_CASE_18.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_18/TEST_CASE_18_TRIAL_0.pfpcw" ]
		then
			printf "TEST CASE 18 FAILED\n"
			printf "\tREASON: Weights file 'TEST_CASE_18_TRIAL_0.pfpcw' was not produced\n"
		else
			printf "TEST CASE 18 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_19 -w PFPC,MFNC > /dev/null 2>&1 )
	then
		printf "TEST CASE 19 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_19" ]
		then
			printf "TEST CASE 19 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_19' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_19/TEST_CASE_19.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_19/TEST_CASE_19.txt" ] 
		then
			printf "TEST CASE 19 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_19.sim' and info file 'TEST_CASE_19.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_19/TEST_CASE_19_TRIAL_0.pfpcw" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_19/TEST_CASE_19_TRIAL_0.mfncw" ]
		then
			printf "TEST CASE 19 FAILED\n"
			printf "\tREASON: Weights files 'TEST_CASE_19_TRIAL_0.pfpcw' and 'TEST_CASE_19_TRIAL_0.mfncw' were not produced\n"
		else
			printf "TEST CASE 19 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	if ! $( $binary -i $workflow_1_input -s $sess_file -o TEST_CASE_20 -w MFNC,PFPC > /dev/null 2>&1 )
	then
		printf "TEST CASE 20 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_20" ]
		then
			printf "TEST CASE 20 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_20' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_20/TEST_CASE_20.sim" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_20/TEST_CASE_20.txt" ] 
		then
			printf "TEST CASE 20 FAILED\n"
			printf "\tREASON: Output simulation 'TEST_CASE_20.sim' and info file 'TEST_CASE_20.txt' were not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_20/TEST_CASE_20_TRIAL_0.pfpcw" ] && \
			 ! [ -e "${data_out_dir}TEST_CASE_20/TEST_CASE_20_TRIAL_0.mfncw" ]
		then
			printf "TEST CASE 20 FAILED\n"
			printf "\tREASON: Weights files 'TEST_CASE_20_TRIAL_0.pfpcw' and 'TEST_CASE_20_TRIAL_0.mfncw' were not produced\n"
		else
			printf "TEST CASE 20 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	
	## invalid test cases
	
	err="$( { $binary -s $sess_file -b $build_file -o TEST_CASE_21 -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify both session and build file. Exiting..." ]]
	then
		printf "TEST CASE 21 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 21 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -b $build_file -i $workflow_1_input -o TEST_CASE_22 -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify both session and build file. Exiting..." ]]
	then
		printf "TEST CASE 22 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 22 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -b $build_file -i $workflow_1_input -o TEST_CASE_23 -r MF,GO -p MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify both session and build file. Exiting..." ]]
	then
		printf "TEST CASE 23 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 23 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -b $build_file -i $workflow_1_input -o TEST_CASE_24 -r MF,GO -p MF,GO -w PFPC,MFNC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Cannot specify both session and build file. Exiting..." ]]
	then
		printf "TEST CASE 24 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 24 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "no input simulation specified in run mode. exiting..." ]]
	then
		printf "TEST CASE 25 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 25 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -o TEST_CASE_26 -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "no input simulation specified in run mode. exiting..." ]]
	then
		printf "TEST CASE 26 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 26 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i NONEXISTENT_FILE.sim -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "Could not find input simulation file 'NONEXISTENT_FILE.sim'. Exiting..." ]]
	then
		printf "TEST CASE 27 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 27 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i NONEXISTENT_FILE.sim -o TEST_CASE_28 -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "Could not find input simulation file 'NONEXISTENT_FILE.sim'. Exiting..." ]]
	then
		printf "TEST CASE 28 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 28 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -r MF,GO > /dev/null; } 2>&1 )"
	if [[ $err =~ "You must specify an output basename. Exiting..." ]]
	then
		printf "TEST CASE 29 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 29 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_30 -r > /dev/null; } 2>&1 )"
	if [[ $err =~ "No parameter given for option '-r'. Exiting..." ]]
	then
		printf "TEST CASE 30 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 30 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_31 -p > /dev/null; } 2>&1 )"
	if [[ $err =~ "No parameter given for option '-p'. Exiting..." ]]
	then
		printf "TEST CASE 31 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 31 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_32 -w > /dev/null; } 2>&1 )"
	if [[ $err =~ "No parameter given for option '-w'. Exiting..." ]]
	then
		printf "TEST CASE 32 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 32 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_33 -r , > /dev/null; } 2>&1 )"
	if [[ $err =~ "Invalid placement of comma for option '-r'. Exiting..." ]]
	then
		printf "TEST CASE 33 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 33 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_34 -p , > /dev/null; } 2>&1 )"
	if [[ $err =~ "Invalid placement of comma for option '-p'. Exiting..." ]]
	then
		printf "TEST CASE 34 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 34 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_35 -w , > /dev/null; } 2>&1 )"
	if [[ $err =~ "Invalid placement of comma for option '-w'. Exiting..." ]]
	then
		printf "TEST CASE 35 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 35 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_36 -r MF,GO,ER > /dev/null; } 2>&1 )"
	if [[ $err =~ "Invalid cell id 'ER' found for option '-r'. Exiting..." ]]
	then
		printf "TEST CASE 36 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 36 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_37 -p MF,GO,ER > /dev/null; } 2>&1 )"
	if [[ $err =~ "Invalid cell id 'ER' found for option '-p'. Exiting..." ]]
	then
		printf "TEST CASE 37 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 37 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_38 -w PFPC,ERRO > /dev/null; } 2>&1 )"
	if [[ $err =~ "Invalid weights id 'ERRO' found for option '-w'. Exiting..." ]]
	then
		printf "TEST CASE 38 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 38 FAILED\n"
	fi
	
	err="$( { $binary --pfpc-off -s $sess_file -i $workflow_1_input -o TEST_CASE_39 --binary > /dev/null; } 2>&1 )"
	if [[ $err =~ "Mutually exclusive or duplicate pfpc plasticity arguments found. Exiting..." ]]
	then
		printf "TEST CASE 39 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 39 FAILED\n"
	fi
	
	err="$( { $binary --binary -s $sess_file -i $workflow_1_input -o TEST_CASE_40 --cascade > /dev/null; } 2>&1 )"
	if [[ $err =~ "Mutually exclusive or duplicate pfpc plasticity arguments found. Exiting..." ]]
	then
		printf "TEST CASE 40 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 40 FAILED\n"
	fi
	
	err="$( { $binary --cascade -s $sess_file -i $workflow_1_input -o TEST_CASE_41 --binary > /dev/null; } 2>&1 )"
	if [[ $err =~ "Mutually exclusive or duplicate pfpc plasticity arguments found. Exiting..." ]]
	then
		printf "TEST CASE 41 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 41 FAILED\n"
	fi
	
	err="$( { $binary --binary -s $sess_file -i $workflow_1_input -o TEST_CASE_42 --binary > /dev/null; } 2>&1 )"
	if [[ $err =~ "Mutually exclusive or duplicate pfpc plasticity arguments found. Exiting..." ]]
	then
		printf "TEST CASE 42 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 42 FAILED\n"
	fi
	
	err="$( { $binary --mfnc-off -s $sess_file -i $workflow_1_input -o TEST_CASE_43 --mfnc-off > /dev/null; } 2>&1 )"
	if [[ $err =~ "Duplicate mfnc plasticity arguments found. Exiting..." ]]
	then
		printf "TEST CASE 43 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 43 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file --session $sess_file -i $workflow_1_input -o TEST_CASE_44 > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-s'. Exiting..." ]]
	then
		printf "TEST CASE 44 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 44 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input --input $workflow_1_input -o TEST_CASE_45 > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-i'. Exiting..." ]]
	then
		printf "TEST CASE 45 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 45 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_46 --output TEST_CASE_46 > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-o'. Exiting..." ]]
	then
		printf "TEST CASE 46 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 46 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_47 -r PC --raster PC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-r'. Exiting..." ]]
	then
		printf "TEST CASE 47 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 47 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_48 -p PC --psth PC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-p'. Exiting..." ]]
	then
		printf "TEST CASE 48 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 48 FAILED\n"
	fi
	
	err="$( { $binary -s $sess_file -i $workflow_1_input -o TEST_CASE_49 -w PFPC --weights PFPC > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-w'. Exiting..." ]]
	then
		printf "TEST CASE 49 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 49 FAILED\n"
	fi
	
	err="$( { $binary -v TUI --visual TUI -s $sess_file -i $workflow_1_input -o TEST_CASE_50 > /dev/null; } 2>&1 )"
	if [[ $err =~ "Found both short and long form of option '-v'. Exiting..." ]]
	then
		printf "TEST CASE 50 PASSED\n"
		(( passed_tests++ ))
	else
		printf "TEST CASE 50 FAILED\n"
	fi
fi

if [[ $# -eq 0 || ( $# -eq 1 && "$1" == "connect" ) ]]; then
	# workflow 2 test cases: collecting connectivity arrays
	## valid test cases
	if ! $binary -b $build_file -o TEST_CASE_51 -c MFGR
	then
		printf "TEST CASE 51 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_51" ]
		then
			printf "TEST CASE 51 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_51' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_51/TEST_CASE_51_PRE.mfgr" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_51/TEST_CASE_51_POST.mfgr" ]
		then
			printf "TEST CASE 51 FAILED\n"
			printf "\tREASON: Pre/Post Output files with base and extension 'TEST_CASE_51.mfgr' were not produced\n"
		else
			printf "TEST CASE 51 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	if ! $binary -b $build_file -o TEST_CASE_52 -c MFGR,MFGO
	then
		printf "TEST CASE 52 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_52" ]
		then
			printf "TEST CASE 52 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_52' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_52/TEST_CASE_52_PRE.mfgr" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_52/TEST_CASE_52_POST.mfgr" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_52/TEST_CASE_52_PRE.mfgo" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_52/TEST_CASE_52_POST.mfgo" ]
		then
			printf "TEST CASE 52 FAILED\n"
			printf "\tREASON: Pre/Post Output files with base and extension 'TEST_CASE_52.mfgr' and 'TEST_CASE_52.mfgo' were not produced\n"
		else
			printf "TEST CASE 52 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	# generating input file for later test cases for workflow 2
	workflow_2_basename="WORKFLOW_2_INPUT"
	workflow_2_input="${workflow_2_basename}.sim"
	if [ -e "${data_out_dir}${workflow_2_basename}/${workflow_2_input}" ]; then
		printf "workflow 2 input sim found. No need to generate a new one...\n"
	else
		printf "Generating simulation for workflow 2 test cases...\n"
		$binary -b $build_file -o $workflow_2_basename
	fi
	if ! $( $binary -i $workflow_2_input -o TEST_CASE_53 -c MFGR > /dev/null 2>&1 )
	then
		printf "TEST CASE 53 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_53" ]
		then
			printf "TEST CASE 53 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_53' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_53/TEST_CASE_53_PRE.mfgr" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_53/TEST_CASE_53_POST.mfgr" ]
		then
			printf "TEST CASE 53 FAILED\n"
			printf "\tREASON: Pre/Post Output files with base and extension 'TEST_CASE_53.mfgr' were not produced\n"
		else
			printf "TEST CASE 53 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
	if ! $( $binary -i $workflow_2_input -o TEST_CASE_54 -c MFGR,MFGO > /dev/null 2>&1 )
	then
		printf "TEST CASE 54 FAILED\n"
		printf "\tREASON: the command returned non-zero exit status\n"
	else
		if ! [ -d "${data_out_dir}TEST_CASE_54" ]
		then
			printf "TEST CASE 54 FAILED\n"
			printf "\tREASON: Output folder 'TEST_CASE_54' was not produced\n"
		elif ! [ -e "${data_out_dir}TEST_CASE_54/TEST_CASE_54_PRE.mfgr" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_54/TEST_CASE_54_POST.mfgr" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_54/TEST_CASE_54_PRE.mfgo" ] && \
		     ! [ -e "${data_out_dir}TEST_CASE_54/TEST_CASE_54_POST.mfgo" ]
		then
			printf "TEST CASE 54 FAILED\n"
			printf "\tREASON: Pre/Post Output files with base and extension 'TEST_CASE_54.mfgr' and 'TEST_CASE_54.mfgo' were not produced\n"
		else
			printf "TEST CASE 54 PASSED\n"
			(( passed_tests++ ))
		fi
	fi
fi

if [ $# -eq 0 ]; then
	printf "All tests finished.\n"
elif [ $# -eq 1 ];then
	if [ "$1" == "build" ]; then
		printf "Build-mode tests finished.\n"
	elif [ "$1" == "run" ]; then
		printf "Run-mode tests finished.\n"
	elif [ "$1" == "connect" ]; then
		printf "Connectivity-mode tests finished.\n"
	fi
fi

printf "${passed_tests}/${num_tests} passed.\n"
cd $data_out_dir
rm -rf TEST_CASE_*
# rm -rf "${workflow_1_basename}"
cd $tests_dir
