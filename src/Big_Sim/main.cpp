#include <getopt.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include "control.h"
#include "gui.h"

const std::string INPUT_DATA_PATH = "../data/inputs/";
const std::string OUTPUT_DATA_PATH = "../data/outputs/";

enum vis_mode {GUI, TUI, NONE};


int main(int argc, char **argv) 
{
	if (argc == 1) 
	{
		std::cerr << "[ERROR]: No file arguments specified. Exiting." << std::endl;
		exit(1);
	}

	enum vis_mode sim_vis_mode = NONE;
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
				if (std::string(optarg) == "gui") sim_vis_mode = GUI;
				else sim_vis_mode = TUI;
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

	switch (sim_vis_mode)
	{
		case GUI:
		{
			exit_status = gui_init_and_run(&argc, &argv);
			break;
		}
		case TUI:
		{
			float mfW = 0.0035; // mf weights (to what?)
			float ws = 0.3275; // weight scale
			float gogr = 0.0105; // gogr weights

			float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
			int grWLength = sizeof(grW) / sizeof(grW[0]);

			std::cout << "[INFO]: Running all simulations..." << std::endl;
			clock_t time = clock();
			for (int goRecipParamNum = 0; goRecipParamNum < 1; goRecipParamNum++)
			{
				float GRGO = grW[goRecipParamNum] * ws; // scaled grgo weights
			   	float MFGO = mfW * ws; // scaled mfgo weights
			   	float GOGR = gogr; // gogr weights, unchanged
			   	for (int simNum = 0; simNum < 1; simNum++)
			   	{
					std::cout << "[INFO]: Running simulation #" << (simNum + 1) << std::endl;
			   		Control control(actParamFile);
					control.runTrials(simNum, GOGR, GRGO, MFGO);
					// TODO: put in output file dir to save to!
					control.saveOutputArraysToFile(goRecipParamNum, simNum);
			   	}
			}
			time = clock() - time;
			std::cout << "[INFO] All simulations finished in "
			   		  << (float) time / CLOCKS_PER_SEC << "s." << std::endl;
			exit_status = 0;
			break;
		}
		case NONE:
		{
			std::cerr << "[INFO] must specify a mode {GUI, TUI}. Exiting..." << std::endl;
			exit_status = 1;
			break;
		}
	}

	return exit_status;
}

