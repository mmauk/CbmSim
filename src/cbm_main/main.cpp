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
	tokenized_file t_e_file;
	lexed_file l_e_file;
	parsed_expt_file e_file;

	tokenize_file(std::string(argv[1]), t_e_file);
	lex_tokenized_file(t_e_file, l_e_file);
	parse_lexed_expt_file(l_e_file, e_file);
	trials_data td = {};
	translate_parsed_trials(e_file, td);
	for (ct_uint32_t i = 0; i < td.num_trials; i++)
	{
		std::cout << td.trial_names[i] << "\n";
	}
	delete_trials_data(td);
	//print_parsed_expt_file(e_file);

	//tokenized_file t_b_file;
	//lexed_file l_b_file;
	//parsed_build_file b_file;
	//tokenize_file(std::string(argv[1]), t_b_file);
	//lex_tokenized_file(t_b_file, l_b_file);
	//parse_lexed_build_file(l_b_file, b_file);
	//print_parsed_build_file(b_file);

	//parsed_commandline p_cl = {};
	//parse_commandline(&argc, &argv, p_cl);
	//print_parsed_commandline(p_cl);
	//validate_commandline(p_cl);
	return 0;
}

int main2(int argc, char **argv) 
{
	parsed_commandline p_cl = {};
	parse_commandline(&argc, &argv, p_cl);
	validate_commandline(p_cl);

	Control *control = new Control(p_cl);
	int exit_status = 0;

	omp_set_num_threads(1); /* for 4 gpus, 8 is the sweet spot. Unsure for 2. */

	if (!p_cl.build_file.empty())
	{
		control->build_sim();
		control->save_sim_to_file(p_cl.output_sim_file);
	}
	else if (!p_cl.experiment_file.empty())
	{
		if (p_cl.vis_mode == "TUI")
		{
			control->runExperiment(NULL);
			if (!p_cl.output_sim_file.empty())
				control->save_sim_to_file(p_cl.output_sim_file);
		}
		else if (p_cl.vis_mode == "GUI")
		{
			exit_status = gui_init_and_run(&argc, &argv, control);
		}
	}
	delete control;
	return exit_status;
}

