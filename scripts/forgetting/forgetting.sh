#!/usr/bin/bash

# grab global vars and functions
source forgetting_setup.sh

set -x

#commence the automation
cd $build_dir

# build the adams (or eves) of bunnies (naming here refers to which ISI will be used down the line)
# for isi in ${isis[@]}; do
#   ./cbm_sim -b forgetting.bld -o forgetting_build_ISI_${isi}
# done

# Step 1: acq and forget data collection
#run_trials_on cbm_sim_gpu_0 eq eq_config
#run_trials_on cbm_sim_gpu_2 acq_no_probe acq_config
#run_trials_on cbm_sim_gpu_2 forget_est_20k forget_est_config # -> might need to run a couple times
run_trials_on cbm_sim_gpu_2 forget_nec_suf forget_nec_suf_config # freeze the weights => dont forget the memory
# run_trials_on cbm_sim_weight_freq forget_data_collect forget_data_collect_config

# Step 2: necessity and sufficiency for CR-mediated PF->PC synapses
# order_reset_insert_weights "acq"

#TODO: need to run the test simulations for each number of reset weights
#      (to figure out how many weights we need to reset)

# run_trials_on cbm_sim_weight_freeze acq acq_nec_suf_config
# run_trials_on cbm_sim_weight_freeze forget forget_nec_suf_config

# Step 3: simulate random walk using external program
# cd $scripts_dir
# ./random_walk.py [frozen_weights_file]
# TODO: may have to run insert_weights.py on the output files from random_walk

# run_trials_on cbm_sim_weight_freq cr_test_random_walk random_walk_config

# Step 4(TODO): forgetting time as a function of gr background activity
# Step 5(TODO): Analysis time baby (faster turn-over times in notebooks at first,
# then for official data analysis include scripts that we can run here :-) )

cd $scripts_dir
