#include <getopt.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "control.h"
#include "gui.h"
#include "fileIO/build_file.h"

enum run_mode {BUILD, RUN, NO_RUN};

void verify_vis_mode(int *argc, char ***argv);
void assign_vis_mode(int *argc, char ***argv, enum vis_mode *sim_vis_mode);
void verify_is_file(int arg_val, char ***argv, std::string error_msg);
void verify_file_format(int arg_val, char ***argv, std::string first_line_test, std::string error_msg);
void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode);

int main(int argc, char **argv)
{
	tokenized_file t_file;
	lexed_file l_file;
	parsed_file p_file;
	std::cout << argv[1] << std::endl;
	tokenize_build_file(std::string(argv[1]), t_file);
	lex_tokenized_build_file(t_file, l_file);
	parse_lexed_build_file(l_file, p_file);

	print_parsed_build_file(p_file);
	return 0;
}

int main2(int argc, char **argv) 
{
// ==================================== PREVIOUS FILE HANDLING ====================================

	enum vis_mode sim_vis_mode = NO_VIS;
	std::string actParamFile = "";
	std::ifstream fileBuf;
	int option;
	while ((option = getopt(argc, argv, "f:m:")) != -1)
	{
		switch (option)
		{
			case 'f':
			{
				actParamFile = INPUT_DATA_PATH + std::string(optarg);
				fileBuf.open(actParamFile.c_str());
				if (!fileBuf)
				{
					fileBuf.close();
					std::cerr << "[ERROR] File " << "'" << std::string(optarg) << "'" 
					          << " does not exist. Exiting." << std::endl;
					exit(2);
				}
				break;
			}
			case 'm':
			{
				if (std::string(optarg) == "gui")
				{
				   sim_vis_mode = GUI;
				} 
				else if (std::string(optarg) == "tui")
				{
					sim_vis_mode = TUI;
				} 
				break;
			}
			case ':':
			{
				std::cerr << "[ERROR] Option needs a value. Exiting." << std::endl;
				exit(3);
				break;
			}
			case '?':
			{
				std::cerr << "[ERROR] Unknown option: " << char(optopt) << ". Exiting." << std::endl;
				exit(4);
				break;
			}
			case -1:
			{
				std::cerr << "[ERROR]: No parameter file given. Exiting." << std::endl;
				exit(5);
				break;
			}
		}
	}

	int exit_status = -1;
	Control *control = new Control(); // default destructor does nothing
	switch (sim_vis_mode)
	{
		case GUI:
		{
			exit_status = gui_init_and_run(&argc, &argv, control);
			break;
		}
		case TUI:
		{
			float mfW = 0.0035; // mf weights (to what?)
			float ws = 0.3275; // weight scale
			float gogr = 0.0105; // gogr weights

			float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
			int grWLength = sizeof(grW) / sizeof(grW[0]);

			std::cout << "[INFO]: Running simulation..." << std::endl;
			clock_t time = clock();
			for (int goRecipParamNum = 0; goRecipParamNum < 1; goRecipParamNum++)
			{
				float GRGO = grW[goRecipParamNum] * ws; // scaled grgo weights
			   	float MFGO = mfW * ws; // scaled mfgo weights
			   	float GOGR = gogr; // gogr weights, unchanged
				control->init_activity_params(actParamFile);
				control->construct_control(TUI);
				control->runTrials(0, GOGR, GRGO, MFGO);
				// TODO: put in output file dir to save to!
				//control->saveOutputArraysToFile(goRecipParamNum, simNum);
			}
			time = clock() - time;
			std::cout << "[INFO] Simulation finished in "
			   		  << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
			exit_status = 0;
			break;
		}
		case NO_VIS:
		{
			std::cerr << "[INFO] must specify a mode {GUI, TUI}. Exiting..." << std::endl;
			exit_status = 1;
			break;
		}
	}
	
	delete control;
	return exit_status;
}

void verify_vis_mode(int *argc, char ***argv)
{
	if (*argv[1] != "tui" && *argv[1] != "gui")
	{
		std::cerr << "[ERROR]: visualization mode {tui, gui} not specified. Exiting..." << std::endl;
		exit(3);
	}
}

void assign_vis_mode(int *argc, char ***argv, enum vis_mode *sim_vis_mode)
{
	if (*argv[1] == "tui") *sim_vis_mode = TUI;
	else if (*argv[1] == "gui") *sim_vis_mode = GUI;
}

void verify_is_file(int arg_val, char ***argv, std::string error_msg)
{
	std::ifstream inFileBuf;
	inFileBuf.open((*argv)[arg_val]);
	if (!inFileBuf.is_open())
	{
		std::cerr << error_msg << std::endl;
		inFileBuf.close();
		exit(4);
	}
	inFileBuf.close();
}

void verify_file_format(int arg_val, char ***argv, std::string first_line_test, std::string error_msg)
{
	std::ifstream inFileBuf;
	inFileBuf.open((*argv)[arg_val]);
	std::string first_line = "";
	getline(inFileBuf, first_line);
	inFileBuf.close();
	if (first_line != first_line_test)
	{
		std::cerr << error_msg << std::endl;
		exit(5);
	}
}

void validate_args_and_set_modes(int *argc, char ***argv,
	  enum vis_mode *sim_vis_mode, enum run_mode *sim_run_mode)
{
	if (*argc >= 4) /* build mode */
	{
		verify_vis_mode(argc, argv);
		assign_vis_mode(argc, argv, sim_vis_mode);

		verify_is_file(2, argv, "[ERROR]: Could not open build file. Exiting...");
		verify_file_format(2, argv, "# BUILD", "[ERROR]: Improper build file format. Exiting...");

		*sim_run_mode = BUILD;
	}
	else if (*argc >= 5) /* run mode */
	{
		verify_vis_mode(argc, argv);
		assign_vis_mode(argc, argv, sim_vis_mode);

		verify_is_file(2, argv, "[ERROR]: Could not open Trials file. Exiting...");
		verify_file_format(2, argv, "# BEGIN EXPERIMENT", "[ERROR] Improper trial file format. Exiting...");

		verify_is_file(3, argv, "[ERROR]: Could not open input simulation file. Exiting...");

		*sim_run_mode = RUN;
	}
	else
	{
		std::cerr << "[ERROR]: Could not parse args. Too few args. Exiting..." << std::endl;
		exit(2);
	}
}
