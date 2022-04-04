

#include <time.h>
#include <iostream>
#include <fstream>

#include <stdDefinitions/pstdint.h>

#include <interfaces/cbmstate.h>
#include <interfaces/iconnectivityparams.h>
#include <interfaces/iactivityparams.h>

#include <interface/cbmsimcore.h>

#include <ecmfpopulation.h>
#include <poissonregencells.h>

#include "control.h"


int **paramArrayPre;
int **paramArray;

int main() 
{

	int numTrainingTrials = 50;
	int homeoTuningTrials = 0;
	int granuleActivityDetectionTrials = 0;
	
	float csTonicF = 0.05;
	float CSlength[3] = {2000, 1000, 1500};
	float goMin = 0.26;

	float GRGO, MFGO, GOGR;
	
	float spillFrac = 0.15;

	float csRateMin = 100.0;
	float csRateMax = 110.0;

	float mfW = 0.0035;
	float ws = 0.3275;
	float gogr = 0.0105;

	float grW[8] = { 0.00056, 0.0007, 0.000933, 0.0014, 0.0028, 0.0056, 0.0112, 0.0224 };
    int grWLength = sizeof(grW) / sizeof(grW[0]);

	float gogoW = 0.0125;

	std::cout << "Starting simulation..." << std::endl;
	clock_t time = clock();
    	
	for (int fileNum = 0; fileNum < 1; fileNum++)
	{	
		std::cout << "ParamNum: " << fileNum << std::endl;
		
		for (int inputWeightNum = 0; inputWeightNum < 1; inputWeightNum++)
		{
			
			for (int goRecipParamNum = 0; goRecipParamNum < grWLength; goRecipParamNum++)
			{
				GRGO = grW[goRecipParamNum] * ws;
				MFGO = mfW * ws;
				GOGR = gogr;

				for (int simNum = 0; simNum < 10; simNum++)
				{
					Control joeSimulation;
					joeSimulation.runSimulationWithGRdata(fileNum, goRecipParamNum, homeoTuningTrials,
							granuleActivityDetectionTrials, numTrainingTrials, simNum, CSlength[fileNum],
							csTonicF, goMin, GOGR, GRGO, MFGO, csRateMin, csRateMax, gogoW, inputWeightNum,
							spillFrac);
				}
			}
		}
	}

	time = clock() - time;
	std::cout << "Simulation completed." << std::endl;
	std::cout << "Total elapsed time: " << (float) time / CLOCKS_PER_SEC << std::endl;
}

