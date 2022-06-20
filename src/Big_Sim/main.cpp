#include <time.h>
#include <iostream>
#include <fstream>
#include "control.h"

const std::string INPUT_DATA_PATH = "../data/inputs/";
const std::string OUTPUT_DATA_PATH = "../data/outputs/";
const std::string ACT_PARAM_FILE = INPUT_DATA_PATH + "actParams.txt";

int main(void) 
{
	float mfW = 0.0035;
	float ws = 0.3275; // weight scale
	float gogr = 0.0105;

	float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
	int grWLength = sizeof(grW) / sizeof(grW[0]);

	std::cout << "[INFO]: Running all simulations..." << std::endl;
	clock_t time = clock();
	for (int goRecipParamNum = 0; goRecipParamNum < grWLength; goRecipParamNum++)
	{
		float GRGO = grW[goRecipParamNum] * ws;
	   	float MFGO = mfW * ws;
	   	float GOGR = gogr; // 
	   	for (int simNum = 0; simNum < 10; simNum++)
	   	{
			std::cout << "[INFO]: Running simulation #" << (simNum + 1) << std::endl;
	   		Control control(ACT_PARAM_FILE);
			control.runTrials(simNum, GOGR, GRGO, MFGO);
			// TODO: put in output file dir to save to!
			control.saveOutputArraysToFile(goRecipParamNum, simNum);
	   	}
	}
	time = clock() - time;
	std::cout << "[INFO] All simulations finished in "
	   		  << (float) time / CLOCKS_PER_SEC << "s." << std::endl;

	return 0;
}

