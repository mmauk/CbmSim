#include <time.h>
#include <iostream>
#include <fstream>
#include "control.h"
#include "gui.h"
#include "commandLine/commandline.h"

int main(int argc, char **argv) 
{
	enum vis_mode sim_vis_mode = NO_VIS;
	enum run_mode sim_run_mode = NO_RUN;
	
	// validates that the args are in the correct format, then sets the vis and run modes
	validate_args_and_set_modes(&argc, &argv, &sim_vis_mode, &sim_run_mode);

	parsed_build_file p_file;
	experiment experiment;
	Control *control = NULL;
	std::string in_sim_file = "";
	std::string out_sim_file = "";
	int exit_status = -1;

	switch (sim_run_mode)
	{
		case BUILD:
			parse_build_args(&argc, &argv, p_file);
			control = new Control();
			control->build_sim(p_file);
			get_out_sim_file(3, &argc, &argv, out_sim_file);
			control->save_sim_to_file(out_sim_file);
			exit_status = 0;
			break;
		case RUN:
			parse_experiment_args(&argc, &argv, experiment); 
			switch (sim_vis_mode)
			{
				case TUI:
					//exit_status = tui_init_and_run(&argc, &argv, control);
					get_in_sim_file(4, &argc, &argv, in_sim_file);
					control = new Control(in_sim_file); 
					control->runExperiment(experiment);
					get_out_sim_file(5, &argc, &argv, out_sim_file);
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

