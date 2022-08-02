/*
 * File: commandline.cpp
 * Author: Sean Gallogly
 * Created on: 08/02/2022
 *
 * Description:
 *     This file implements the function prototypes in commandLine/commandline.h
 */

#include <iostream>
#include <fstream>
#include <algorithm>
#include "commandLine/commandline.h"

/*
 * Implementation Notes:
 *     Attempts to open the file via an input file stream. Upon failure,
 *     returns false and on success returns true. Assumes that the raw
 *     argument given is the file basename and that the full path is that
 *     given by INPUT_DATA_PATH. 
 *
 *     TODO: let INPUT_DATA_PATH be an argument so that we can also check
 *           output files.
 *
 */
bool is_file(const char *arg)
{
	std::ifstream inFileBuf;
	std::string full_in_file_path = INPUT_DATA_PATH + std::string(arg);
	inFileBuf.open(full_in_file_path.c_str());
	if (!inFileBuf.is_open()) return false;
	inFileBuf.close();
	return true;
}

/*
 * Implementation Notes:
 *     Attempts to open the file via an input file stream. Upon failure, simply
 *     returns the "null" case for the run_mode enum, which is NO_RUN. Otherwise,
 *     checks whether the first line of the file matches that of a predefined,
 *     named constant for the build file or that of the experiment file.
 *
 *     TODO: Allow more flexibility in either input file type by searching for
 *           the named string constants rather than assuming they will be the
 *           first lines of each file type.
 *
 */
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

/*
 * Implementation Notes:
 *     Attempts to open the file via an input file stream. Upon failure simply
 *     returns the "null" case for the vis_mode enum, which is NO_VIS. Otherwise,
 *     checks whether the second line of the file matches either '#VIS TUI' or
 *     '#VIS GUI', and returns the respective vis_mode enum.
 *
 *     TODO: Allow more flexibility in either input file by searching for these 
 *           strings before some invocation of '#begin' instead of assuming
 *           they will occur at the second line.
 *
 */
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

/*
 * Implementation Notes:
 *     Does a single pass over all command line args, then establishes whether in an error state
 *     due to inadherence of args to two-mode scheme specified in commandline.h. There are a few
 *     possible error states:
 *
 *         1) a build file was specified but there were either too few or too many args for that mode.
 *         2) a build or experiment file was specified, but is not in the correct format.
 *         3) an experiment file was specified but there were either too few or too many args for that mode.
 *         4) an experiment file was specified, but the visualization mode was not specified in the second line
 *         5) not build nor experiment file was specified, resulting in an unknown run state.
 *
 *     For 2) above, we assume either that the build file or the experiment file is in the correct format if
 *     the first line is valid. We leave the error checking on those files to the parsing functions for each.
 *
 *     TODO: create an enum of the possible error states that we can find ourselves in, use as argument to exit
 *           so that we can debug from controlling bash script if we have printing to some buffer disabled.
 *
 */
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

/*
 * Implementation Notes:
 *     parses an input file, interpreted as a build file, from somewhere in argv.
 *     This function acts to bundle up the parsing process for the build file into a single function.
 *     For more information on the steps to process a build_file into a parsed_build_file structure,
 *     see build_file.h and build_file.cpp
 *
 */
void parse_build_args(char ***argv, parsed_build_file &p_file)
{
	tokenized_file t_file;
	lexed_file l_file;

	std::string full_build_file_path = INPUT_DATA_PATH + std::string((*argv)[BUILD_FILE]);

	tokenize_build_file(full_build_file_path, t_file);
	lex_tokenized_build_file(t_file, l_file);
	parse_lexed_build_file(l_file, p_file);
}

/*
 * Implementation Notes:
 *     parses an input file, interpreted as an experiment file, from somewhere in argv.
 *     This function essentially 'wraps' parse_experiment_file by allowing the user to
 *     only supply the address of the argv array, instead of having to specify which argument
 *     is supposed to be the experiment file, and prepending the input data path to the raw argument.
 *     This is done as a matter of expediency and taste and can be removed in favour of calling
 *     parse_experiment_file directly.
 *
 */
void parse_experiment_args(char ***argv, experiment &exper)
{
	std::string full_trial_file_path = INPUT_DATA_PATH + std::string((*argv)[EXPERIMENT_FILE]);
	parse_experiment_file(full_trial_file_path, exper);
}

/*
 * Implementation Notes:
 *     gets the input simulation file from argv by prepending the full input file path to it.
 *
 */
void get_in_sim_file(char ***argv, std::string &in_file)
{
	in_file = INPUT_DATA_PATH + std::string((*argv)[IN_SIM_FILE]);
}

/*
 * Implementation Notes:
 *     gets the output simulation file from argv by prepending the full input data path to it.
 *     Notice that we're prepending the *input* data path to the *output* sim file. We do this
 *     for now to facillitate daisy-chaining of simulations together, but we could make this more
 *     flexible by specifying a file path argument as well. We also include the argument index,
 *     as it varies by one depending upon whether we are in build mode or run mode.
 */

void get_out_sim_file(int arg_index, char ***argv, std::string &out_file)
{
	out_file = INPUT_DATA_PATH + std::string((*argv)[arg_index]);
}

