#include <getopt.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "control.h"
#include "gui.h"
#include "fileIO/build_file.h"

void verify_and_assign_run_mode(int arg_index, int *argc, char ***argv, enum run_mode *sim_run_mode);
void verify_vis_mode(int arg_index, int *argc, char ***argv);
void assign_vis_mode(int arg_index, int *argc, char ***argv, enum vis_mode *sim_vis_mode);
void verify_is_file(int arg_index, char ***argv, std::string error_msg);
void verify_file_format(int arg_index, char ***argv, std::string first_line_test, std::string error_msg);
void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode);

void parse_build_args(int *argc, char ***argv, parsed_file &p_file);
void get_out_sim_file(int arg_index, int *argc, char ***argv, std::string &out_file);

int main1(int argc, char **argv)
{
	tokenized_file t_file;
	lexed_file l_file;
	parsed_file p_file;

	tokenize_build_file(std::string(argv[1]), t_file);
	lex_tokenized_build_file(t_file, l_file);
	parse_lexed_build_file(l_file, p_file);

	ConnectivityParams cp(p_file);
	ActivityParams ap(p_file);
	std::cout << cp << std::endl;
	std::cout << ap << std::endl;
	//print_lexed_build_file(l_file);
	//print_parsed_build_file(p_file);
	return 0;
}

int main(int argc, char **argv) 
{
// ==================================== PREVIOUS FILE HANDLING ====================================

	enum vis_mode sim_vis_mode = NO_VIS;
	enum run_mode sim_run_mode = NO_RUN;
	
	// validates that the args are in the correct format, then sets the vis and run modes
	validate_args_and_set_modes(&argc, &argv, &sim_vis_mode, &sim_run_mode);

	parsed_file p_file;
	Control *control = NULL;
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
			//parse_run_args(&argc, &argv) // TODO: write this, leverage mike's parse alg
			switch (sim_vis_mode)
			{
				case TUI:
					//exit_status = tui_init_and_run(&argc, &argv, control);
					break;
				case GUI:
					exit_status = gui_init_and_run(&argc, &argv, control);
					break;
			}
			break;
		case NO_RUN:
			/* unreachable */
			break;
	}

	//Control *control = new Control(); // default destructor does nothing
	//switch (sim_vis_mode)
	//{
	//	case GUI:
	//	{
	//		exit_status = gui_init_and_run(&argc, &argv, control);
	//		break;
	//	}
	//	case TUI:
	//	{
	//		float mfW = 0.0035; // mf weights (to what?)
	//		float ws = 0.3275; // weight scale
	//		float gogr = 0.0105; // gogr weights

	//		float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
	//		int grWLength = sizeof(grW) / sizeof(grW[0]);

	//		std::cout << "[INFO]: Running simulation..." << std::endl;
	//		clock_t time = clock();
	//		for (int goRecipParamNum = 0; goRecipParamNum < 1; goRecipParamNum++)
	//		{
	//			float GRGO = grW[goRecipParamNum] * ws; // scaled grgo weights
	//		   	float MFGO = mfW * ws; // scaled mfgo weights
	//		   	float GOGR = gogr; // gogr weights, unchanged
	//			control->init_activity_params(actParamFile);
	//			control->construct_control(TUI);
	//			control->runTrials(0, GOGR, GRGO, MFGO);
	//			// TODO: put in output file dir to save to!
	//			//control->saveOutputArraysToFile(goRecipParamNum, simNum);
	//		}
	//		time = clock() - time;
	//		std::cout << "[INFO] Simulation finished in "
	//		   		  << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
	//		exit_status = 0;
	//		break;
	//	}
	//	case NO_VIS:
	//	{
	//		std::cerr << "[INFO] must specify a mode {GUI, TUI}. Exiting..." << std::endl;
	//		exit_status = 1;
	//		break;
	//	}
	//}
	
	delete control;
	return exit_status;
}

void verify_and_assign_run_mode(int arg_index, int *argc, char ***argv, enum run_mode *sim_run_mode)
{
	if (*argc >= 2)
	{
		if (std::string((*argv)[arg_index]) == "build") *sim_run_mode = BUILD;
		else if (std::string((*argv)[arg_index]) == "run") *sim_run_mode = RUN;
		else
		{
			std::cerr << "[ERROR]: Invalid run mode specified. Choices are: 'build' or 'run'"
					  << std::endl;
			exit(arg_index);
		}
	}
	else
	{
		std::cerr << "[ERROR]: Not enough arguments. Expected a run mode {build, run} as the first argument. Exiting..."
				  << std::endl;
		exit(arg_index);
	}
}

void verify_vis_mode(int arg_index, int *argc, char ***argv)
{
	if (std::string((*argv)[arg_index]) != "tui" && std::string((*argv)[arg_index]) != "gui")
	{
		std::cerr << "[ERROR]: visualization mode {tui, gui} not specified. Exiting..." << std::endl;
		exit(arg_index);
	}
}

void assign_vis_mode(int arg_index, int *argc, char ***argv, enum vis_mode *sim_vis_mode)
{
	if (std::string((*argv)[arg_index]) == "tui") *sim_vis_mode = TUI;
	else if (std::string((*argv)[arg_index]) == "gui") *sim_vis_mode = GUI;
}

void verify_is_file(int arg_index, char ***argv, std::string error_msg)
{
	std::ifstream inFileBuf;
	std::string full_in_file_path = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
	inFileBuf.open(full_in_file_path.c_str());
	if (!inFileBuf.is_open())
	{
		std::cerr << error_msg << std::endl;
		inFileBuf.close();
		exit(arg_index);
	}
	inFileBuf.close();
}

void verify_file_format(int arg_index, char ***argv, std::string first_line_test, std::string error_msg)
{
	std::string full_build_file_path = INPUT_DATA_PATH + std::string((*argv)[2]);
	std::ifstream inFileBuf;
	inFileBuf.open(full_build_file_path.c_str());
	std::string first_line = "";
	getline(inFileBuf, first_line);
	inFileBuf.close();
	if (first_line != first_line_test)
	{
		std::cerr << error_msg << std::endl;
		exit(arg_index);
	}
}

void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode)
{
	verify_and_assign_run_mode(1, argc, argv, sim_run_mode);

	if (*sim_run_mode == BUILD && *argc <= 4) /* build mode */
	{
		verify_is_file(2, argv, "[ERROR]: Could not open build file. Exiting...");
		verify_file_format(2, argv, "#begin filetype build", "[ERROR]: Improper build file format. Exiting...");
	}
	else if (*sim_run_mode == RUN && *argc <= 6) /* run mode */
	{
		verify_vis_mode(2, argc, argv);
		assign_vis_mode(2, argc, argv, sim_vis_mode);

		verify_is_file(3, argv, "[ERROR]: Could not open Trials file. Exiting...");
		verify_file_format(3, argv, "# BEGIN EXPERIMENT", "[ERROR] Improper trial file format. Exiting...");

		verify_is_file(4, argv, "[ERROR]: Could not open input simulation file. Exiting...");
	}
	else
	{
		std::cerr << "[ERROR]: Could not parse args. Too many args. Exiting..." << std::endl;
		exit(-1);
	}
}

void parse_build_args(int *argc, char ***argv, parsed_file &p_file)
{
	tokenized_file t_file;
	lexed_file l_file;

	std::string full_build_file_path = INPUT_DATA_PATH + std::string((*argv)[2]);

	tokenize_build_file(full_build_file_path, t_file);
	lex_tokenized_build_file(t_file, l_file);
	parse_lexed_build_file(l_file, p_file);
}

void get_out_sim_file(int arg_index, int *argc, char ***argv, std::string &out_file)
{
	// check whether the arg index we provided is equal to the last available command line index
	if (*argc != arg_index + 1) out_file = DEFAULT_SIM_OUT_FILE;
	else out_file = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
}

