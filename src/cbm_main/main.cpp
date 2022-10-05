/*
 * File: main.cpp
 * Author: Sean Gallogly
 * Created on: circa 07/21/2022
 * 
 * Description:
 *     this is the main entry point to the program. It calls functions from commandline.h
 *     in order to parse arguments and from control.h in order to run the simulation
 *     in one of several user-specified modes.
 *
 */

#include <time.h>
#include <iostream>
#include <fstream>
#include <omp.h>

#include "control.h"
#include "gui.h"
#include "commandline.h"
#include "activityparams.h"
#include "file_parse.h"

int main(int argc, char **argv)
{
	//tokenized_file t_e_file;
	//lexed_file l_e_file;
	//parsed_expt_file e_file;

	//tokenize_file(std::string(argv[1]), t_e_file);
	//lex_tokenized_file(t_e_file, l_e_file);
	//parse_lexed_expt_file(l_e_file, e_file);
	//print_parsed_expt_file(e_file);

	//tokenized_file t_b_file;
	//lexed_file l_b_file;
	//parsed_build_file b_file;
	//tokenize_file(std::string(argv[1]), t_b_file);
	//lex_tokenized_file(t_b_file, l_b_file);
	//parse_lexed_build_file(l_b_file, b_file);
	//print_parsed_build_file(b_file);

	parsed_commandline p_cl = {};
	parse_commandline(&argc, &argv, p_cl);
	print_parsed_commandline(p_cl);
	return 0;
}

int main2(int argc, char **argv) 
{
	enum vis_mode sim_vis_mode  = NO_VIS;
	enum run_mode sim_run_mode  = NO_RUN;
	enum user_mode sim_user_mode = NO_USER_MODE;
	
	validate_args_and_set_modes(&argc, &argv, &sim_vis_mode, &sim_run_mode, &sim_user_mode);

	parsed_build_file p_file;
	Control *control = NULL;
	std::string out_sim_file = "";
	int exit_status = -1;

	omp_set_num_threads(1); /* for 4 gpus, 8 is the sweet spot. Unsure for 2. */

	switch (sim_run_mode)
	{
		case BUILD: /* parse build file, build sim, save to file and exit */
			parse_build_args(&argv, p_file);
			control = new Control(p_file);
			control->build_sim();
			get_out_sim_file(BUILD_OUT_SIM_FILE, &argv, out_sim_file);
			control->save_sim_to_file(out_sim_file);
			exit_status = 0;
			break;
		case RUN:
			switch (sim_user_mode)
			{
				case FRIENDLY: /* you have to load everything in yourself in gui, but you only enter the command on cmdline */
					control = new Control(GUI);
					break;
				case VETERAN: /* you have to know the name of every input file when invoking on cmdline */
					control = new Control(&argv, sim_vis_mode); 
					break;
				case NO_USER_MODE:
					/* unreachable */
					break;
			}
			switch (sim_vis_mode)
			{
				case TUI:
					control->runExperiment(NULL);
					//get_out_sim_file(RUN_OUT_SIM_FILE, &argv, out_sim_file);
					//control->save_sim_to_file(out_sim_file);
					exit_status = 0;
					break;
				case GUI:
					exit_status = gui_init_and_run(&argc, &argv, control);
					break;
				case NO_VIS:
					/* unreachable */
					break;
			}
			break;
		case NO_RUN:
			/* unreachable */
			break;
	}

	delete control;
	return exit_status;
}

