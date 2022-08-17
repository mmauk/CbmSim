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
#include "control.h"
#include "gui.h"
#include "commandLine/commandline.h"
#include "params/activityparams.h"

int main1(int argc, char **argv)
{
	parsed_build_file p_file;
	parse_build_args(&argv, p_file);
	populate_con_params(p_file);
	populate_act_params(p_file);
	return 0;
}


int main(int argc, char **argv) 
{
	enum vis_mode sim_vis_mode = NO_VIS;
	enum run_mode sim_run_mode = NO_RUN;
	
	validate_args_and_set_modes(&argc, &argv, &sim_vis_mode, &sim_run_mode);

	parsed_build_file p_file;
	experiment experiment;
	Control *control = NULL;
	//std::string in_sim_file = "";
	std::string out_sim_file = "";
	int exit_status = -1;

	switch (sim_run_mode)
	{
		case BUILD:
			parse_build_args(&argv, p_file);
			control = new Control();
			control->build_sim(p_file);
			get_out_sim_file(BUILD_OUT_SIM_FILE, &argv, out_sim_file);
			control->save_sim_to_file(out_sim_file);
			exit_status = 0;
			break;
		case RUN:
			control = new Control(&argv, sim_vis_mode); 
			switch (sim_vis_mode)
			{
				case TUI:
					control->runTrials(0, 0, 0, 0);
					//control->runExperiment(experiment);
					get_out_sim_file(RUN_OUT_SIM_FILE, &argv, out_sim_file);
					control->save_sim_to_file(out_sim_file);
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

