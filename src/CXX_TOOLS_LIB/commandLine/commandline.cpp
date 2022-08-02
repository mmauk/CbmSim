#include <iostream>
#include <fstream>
#include <algorithm>
#include "commandLine/commandline.h"

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
	std::string full_build_file_path = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
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
		verify_file_format(3, argv, "#Begin Experiment", "[ERROR] Improper trial file format. Exiting...");

		verify_is_file(4, argv, "[ERROR]: Could not open input simulation file. Exiting...");
	}
	else
	{
		std::cerr << "[ERROR]: Could not parse args. Too many args. Exiting..." << std::endl;
		exit(-1);
	}
}

void parse_build_args(int *argc, char ***argv, parsed_build_file &p_file)
{
	tokenized_file t_file;
	lexed_file l_file;

	std::string full_build_file_path = INPUT_DATA_PATH + std::string((*argv)[2]);

	tokenize_build_file(full_build_file_path, t_file);
	lex_tokenized_build_file(t_file, l_file);
	parse_lexed_build_file(l_file, p_file);
}

void parse_experiment_args(int *argc, char ***argv, experiment &exper)
{
	std::string full_trial_file_path = INPUT_DATA_PATH + std::string((*argv)[3]);
	parse_experiment_file(full_trial_file_path, exper);
}

void get_in_sim_file(int arg_index, int *argc, char ***argv, std::string &in_file)
{
	in_file = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
}

void get_out_sim_file(int arg_index, int *argc, char ***argv, std::string &out_file)
{
	// check whether the arg index we provided is equal to the last available command line index
	if (*argc != arg_index + 1) out_file = DEFAULT_SIM_OUT_FILE;
	else out_file = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
}

char* getCmdOption(char** begin, char** end, const std::string& option)
{
	char** itr = std::find(begin, end, option);
	if (itr != end && ++itr !=end)
	{
		return *itr;
	}
	return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}


