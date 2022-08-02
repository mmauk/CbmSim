#include <iostream>
#include <fstream>
#include <algorithm>
#include "commandLine/commandline.h"

bool is_file(const char *arg)
{
	std::ifstream inFileBuf;
	std::string full_in_file_path = INPUT_DATA_PATH + std::string(arg);
	inFileBuf.open(full_in_file_path.c_str());
	if (!inFileBuf.is_open()) return false;
	inFileBuf.close();
	return true;
}

enum run_mode get_run_mode(const char *arg)
{
	std::ifstream inFileBuf;
	std::string full_in_file_path = INPUT_DATA_PATH + std::string(arg);
	inFileBuf.open(full_in_file_path.c_str());
	std::string first_line;
	getline(inFileBuf, first_line);
	inFileBuf.close();
	if (first_line == BUILD_FILE_FIRST_LINE)
	{
		return BUILD;
	}
	else if (first_line == EXPERIMENT_FILE_FIRST_LINE)
	{
		return RUN;
	}
	else return NO_RUN;
}

enum vis_mode get_vis_mode(const char *arg)
{
	std::ifstream inFileBuf;
	std::string full_in_file_path = INPUT_DATA_PATH + std::string(arg);
	inFileBuf.open(full_in_file_path.c_str());
	std::string first_line, second_line;
	getline(inFileBuf, first_line);
	getline(inFileBuf, second_line);
	inFileBuf.close();
	if (second_line == "#VIS TUI") return TUI;
	else if (second_line == "#VIS GUI") return GUI;
	else return NO_VIS;
}

void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode)
{
	for (int i = 1; i < *argc; i++)
	{
		if (is_file((*argv)[i]))
		{
			switch (get_run_mode((*argv)[i]))
			{
				case BUILD:
					*sim_run_mode = BUILD;
					break;
				case RUN:
					*sim_run_mode = RUN;
					switch (get_vis_mode((*argv)[i]))
					{
						case TUI:    {*sim_vis_mode = TUI; break;}
						case GUI:    {*sim_vis_mode = GUI; break;}
						case NO_VIS: {break;}
					}
					break;
				case NO_RUN:
					break;
			}
		}
	}

	switch (*sim_run_mode)
	{
		case BUILD:
			if (*argc != BUILD_NUM_ARGS)
			{
				std::cerr << "[ERROR]: Obtained build file but could not obtain output sim file. Exiting..."
						  << std::endl;
				exit(-1);
			}
			break;
		case RUN:
			if (*argc != RUN_NUM_ARGS)
			{
				std::cerr << "[ERROR]: Obtained experiment file but could not obtain input or output sim file. Exiting..."
						  << std::endl;
				exit(-1);
			}
			else
			{
				if (*sim_vis_mode == NO_VIS)
				{
					std::cerr << "[ERROR]: Obtained experiment file but no visualization mode was specified in that file."
							  << std::endl;
					std::cerr << "[ERROR]: (Hint: did you add a second line saying '#VIS {TUI, GUI}'?"
							  << std::endl;
					std::cerr << "[ERROR]: Exiting..." << std::endl;
					exit(-1);
				}
				if (!is_file((*argv)[IN_SIM_FILE]))
				{
					std::cerr << "[ERROR]: Simulation file argument specified but is not a valid file."
							  << std::endl;
					std::cerr << "[ERROR]: Exiting..." << std::endl;
					exit(-1);
				}
			}
			break;
		case NO_RUN:
			std::cerr << "[ERROR]: Could not determine which mode we are running in." << std::endl;
			std::cerr << "[ERROR]: (Hint: did you include a valid build or experiment file?)" << std::endl;
			std::cerr << "[ERROR]: Exiting..." << std::endl;
			exit(-1);
			break;
	}
}

void parse_build_args(char ***argv, parsed_build_file &p_file)
{
	tokenized_file t_file;
	lexed_file l_file;

	std::string full_build_file_path = INPUT_DATA_PATH + std::string((*argv)[BUILD_FILE]);

	tokenize_build_file(full_build_file_path, t_file);
	lex_tokenized_build_file(t_file, l_file);
	parse_lexed_build_file(l_file, p_file);
}

void parse_experiment_args(char ***argv, experiment &exper)
{
	std::string full_trial_file_path = INPUT_DATA_PATH + std::string((*argv)[EXPERIMENT_FILE]);
	parse_experiment_file(full_trial_file_path, exper);
}

void get_in_sim_file(char ***argv, std::string &in_file)
{
	in_file = INPUT_DATA_PATH + std::string((*argv)[IN_SIM_FILE]);
}

void get_out_sim_file(int arg_index, char ***argv, std::string &out_file)
{
	out_file = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
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


